from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import os
import time
import psutil
from copy import deepcopy
import torch as th
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor, should_collect_more_steps
from collections import deque
from .rp_agents import DQNPRM
from .commons_agent import DQN
from ..utils import DummyGymEnv

class IndependentDQNRP(DQN):
    def __init__(self, env: Union[GymEnv, str], num_agents: int, predictor, *args, **kwargs):
        super().__init__(env=env, *args, **kwargs)
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        self.predictor = predictor
        self.loggers = []
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.agents = [DQNPRM(env=dummy_env, predictor=None, *args, **kwargs) for _ in range(self.num_agents)]
    
    def _setup_learn(self,  total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar):
       total_timesteps, callback  = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar
        )
       for agent in self.agents:
            if agent.ep_info_buffer is None or reset_num_timesteps:
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                agent.ep_info_buffer = deque(maxlen=self._stats_window_size)
                agent.ep_success_buffer = deque(maxlen=self._stats_window_size)

       return  total_timesteps, callback

    def update_agents_last_obs(self):
        for i in range(self.num_agents):
            last_obs = self._last_obs[i::self.num_agents]
            self.agents[i]._last_obs = last_obs

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for agent in self.agents:
            agent.train(gradient_steps, batch_size)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        actions = np.zeros((self.num_envs * self.num_agents))
        for i, agent in enumerate(self.agents):
            ac, _ = agent.predict(
                observation[i::self.num_agents],
                state,
                episode_start,
                deterministic
            )
            actions[i::self.num_agents] = ac
        return actions, None

    def set_logger(self, loggers):
        super().set_logger(loggers[0])
        for i, agent in enumerate(self.agents):
           agent.set_logger(loggers[i])
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffers: List[ReplayBuffer],
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        for agent in self.agents:
            agent.policy.set_training_mode(False)

        n_collected_steps, n_collected_episodes = 0, 0

        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, n_collected_steps, n_collected_episodes):
            agents_actions, agents_buffer_actions = [], []
            for agent in self.agents:
                actions_, buffer_actions_ = agent._sample_action(learning_starts, action_noise, self.num_envs)
                agents_actions.append(actions_)
                agents_buffer_actions.append(buffer_actions_)

            actions = np.concatenate(np.array(agents_actions).T, axis=0)
            buffer_actions = np.concatenate(np.array(agents_buffer_actions).T, axis=0)

            new_obs, expiriment_rewards, dones, infos = env.step(actions)

            pred_rewards = self.predictor.predict(
                obs_as_tensor(self._last_obs, self.agents[0].policy.device),
                th.tensor(actions).to(self.agents[0].policy.device)
            ).squeeze()

            self.num_timesteps += env.num_envs
            for agent in self.agents:
                agent.num_timesteps += env.num_envs
            n_collected_steps += 1

            callback.update_locals(locals())

            if callback.on_step() is False:
                return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training=False)
            
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            for i, agent in enumerate(self.agents):
                agent._update_info_buffer(infos[i::self.num_agents],
                                          dones[i::self.num_agents])

            self._last_obs = new_obs

            for i, (agent, replay_buffer) in enumerate(zip(self.agents, replay_buffers)):
                agent._store_transition(
                    replay_buffer,
                    buffer_actions[i::self.num_agents],
                    new_obs[i::self.num_agents],
                    pred_rewards[i::self.num_agents],
                    expiriment_rewards[i::self.num_agents],
                    dones[i::self.num_agents],
                    infos[i::self.num_agents],
                )

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
                        for agent in self.agents:
                            agent._dump_logs()
                        super()._dump_logs()


                    
        callback.on_rollout_end()

        return RolloutReturn(n_collected_steps * env.num_envs, n_collected_episodes, continue_training)  