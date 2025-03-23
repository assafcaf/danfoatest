from typing import Any, Dict, List, Optional, Type, Union
from ..utils import DummyGymEnv
import time
import numpy as np
import os
from copy import deepcopy
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
import psutil 
from .single_agent import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import functional as F




class DQNPRM(DQN):
    def __init__(self,
                predictor,
                env:Union[GymEnv, str],
                *args, **kwargs):
        
        super().__init__(env=env, *args, **kwargs)
        self.predictor = predictor

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, expiriment_rewards, dones, infos = env.step(actions)
            # reward predictor
            pred_rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device),
                                    th.tensor(actions).to(self.policy.device)).squeeze()

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer,
                buffer_actions,
                new_obs,
                pred_rewards,
                expiriment_rewards,
                dones,
                infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                    
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        expiriment_rewards: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            expiriment_rewards,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    @property
    def buffer(self):
        return self.replay_buffer

class DQNCRM(DQN):
    def __init__(self, env: Union[GymEnv, str], predictor, num_agents: int, *args, **kwargs):
        super().__init__(env=env, *args, **kwargs)
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents

        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        kwargs['buffer_size'] = 0
        self.agents = [DQN(env=dummy_env, *args, **kwargs) for _ in range(self.num_agents)]
        self.predictor = predictor

    def update_agents_last_obs(self):
        for agent_id in range(self.num_agents):
            last_obs = self._last_obs[agent_id::self.num_agents]
            self.agents[agent_id]._last_obs = last_obs

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        log_interval: int = 1
    ) -> "OffPolicyAlgorithm":
    
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar
        )
        self.update_agents_last_obs()
        for agent in self.agents:
            agent.set_logger([self.logger])

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.agents[0].train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=[agent.replay_buffer for agent in self.agents],
                log_interval=log_interval,
            )
            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            
            # check memory usage and if it exceeds 80% shut down the training 
            if psutil.virtual_memory().percent > 80:
                print("Memory usage is above 80%. Stopping training.")
                exit()
            
          
        callback.on_training_end()

        return self
           
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: List[ReplayBuffer],
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        for agent in self.agents:
            agent.policy.set_training_mode(False)

        n_collected_steps, n_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            for agent in self.agents:
                agent.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, n_collected_steps, n_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and n_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for agent in self.agents:
                    agent.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            agents_actions, agents_buffer_actions = [], []
            for agent in self.agents:
                actions_, buffer_actions_ = agent._sample_action(learning_starts, action_noise, self.num_envs)
                agents_actions.append(actions_)
                agents_buffer_actions.append(buffer_actions_)

            #  concatenate actions of all agents
            actions = np.concatenate(np.array(agents_actions).T, axis=0)
            buffer_actions = np.concatenate(np.array(agents_buffer_actions).T, axis=0)
            

            new_obs, expiriment_rewards, dones, infos = env.step(actions)
            # reward predictor
            states = np.array(env.get_full_states())
            pred_rewards =  self.predictor.predict(obs_as_tensor(states, self.policy.device),
                                                   th.tensor(actions).to(self.policy.device)).flatten()
            
            self.num_timesteps += env.num_envs
            for agent in self.agents:
                agent.num_timesteps += env.num_envs
            n_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer,
                buffer_actions,
                new_obs,
                pred_rewards,
                expiriment_rewards,
                states,
                dones,
                infos)
            
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            for agent in self.agents:
                 agent._update_current_progress_remaining(agent.num_timesteps, self._total_timesteps)
            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()
            for agent in self.agents:
                 agent._on_step()
                 
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    n_collected_episodes += 1
                    self._episode_num += 1
                    for agent in self.agents:
                        agent._episode_num += 1
                    

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if log_interval is not None and self._episode_num % log_interval == 0:
                        super()._dump_logs()


                    
        callback.on_rollout_end()

        return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training)  

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        expiriment_rewards: np.ndarray,
        states: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            expiriment_rewards,
            states,
            dones,
            infos,
        )

        self._last_obs = new_obs
        self.update_agents_last_obs()
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def train_multiagent(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Multi-agent version of the DQN train function.
        Each agent is trained independently using its own replay buffer.
        
        :param gradient_steps: Number of gradient steps to perform.
        :param batch_size: Batch size for each agent's training step.
        """
        for agent_id, agent in enumerate(self.agents):  # Assume self.agents is a list of DQN agents
            # Switch to train mode for the current agent
            agent.policy.set_training_mode(True)
            # Update learning rate for the current agent
            agent._update_learning_rate(agent.policy.optimizer)
            
            losses = []
            for _ in range(gradient_steps):
                # Sample replay buffer for the current agent
                replay_data = self.replay_buffer.sample(batch_size, env=agent._vec_normalize_env, agent_id=agent_id)  # type: ignore[union-attr]

                with th.no_grad():
                    # Compute the next Q-values using the target network
                    next_q_values = agent.q_net_target(replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    # 1-step TD target
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * agent.gamma * next_q_values

                # Get current Q-values estimates
                current_q_values = agent.q_net(replay_data.observations)

                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())

                # Optimize the policy for the current agent
                agent.policy.optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
                agent.policy.optimizer.step()

            # Increase update counter for the current agent
            agent._n_updates += gradient_steps
            agent.logger.record(f"agent_{agent_id}/train/n_updates", agent._n_updates, exclude="tensorboard")  # Assuming agent has a unique id
            agent.logger.record(f"agent_{agent_id}/train/loss", np.mean(losses))

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[DQNPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ):
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.policies[polid] = sb3_DQN.load(
                path=path + f"/policy_{polid + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for agents_id in range(self.num_agents):
            os.makedirs(path + f"/agent_{agents_id + 1}/", exist_ok=True)
            self.agents[agents_id].save(path=path + f"/agent_{agents_id + 1}/model")
    
    def set_up_agents(self):
        for agent in self.agents:
            agent.start_time = time.time()

            if agent.action_noise is not None:
                agent.action_noise.reset()

            agent.set_logger(self.logger)
            agent._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

    def set_logger(self, logger):
        super().set_logger(logger)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        actions = np.zeros((self.num_envs * self.num_agents))
        for agent_id, agent in enumerate(self.agents):
            ac, _ = agent.predict(
                observation[agent_id::self.num_agents],
                state,
                episode_start,
                deterministic
            )
            actions[agent_id::self.num_agents] = ac
        return actions, None


    def get_buffers(self):
        return [agent.replay_buffer for agent in self.agents]