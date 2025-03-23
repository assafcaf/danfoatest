from stable_baselines3 import PPO as sb3_PPO
import numpy as np
import gymnasium
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import time
from collections import deque
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger, utils 
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import psutil 
import asyncio

from multiprocessing import Process, Queue
from. single_agent import PPO

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
class IndependentPPO(PPO):
    def __init__(self, env: Union[GymEnv, str], num_agents: int, *args, **kwargs):
        super().__init__(env=env, *args, **kwargs)
        
        self.num_agents = num_agents
        self.num_envs = env.num_envs // num_agents
        
        # create empty vectiruze enc with apropriate params such as num_envs
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        
        self.agents = [PPO(env=dummy_env, *args, **kwargs) for _ in range(self.num_agents)]

        
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[List[MaybeCallback]] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
        log_interval: int = 4,
    ):
        iteration = 0
        # init agent params
        total_timesteps, callback = self._setup_learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=tb_log_name,
                progress_bar=progress_bar
            )
        
        self.update_agents_last_obs()
        for agent in self.agents:
            agent.set_logger([self.logger])
        # init agents 

        # # TODO: do I need this?
        # self.update_last_episode_starts()
                
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        
        # collect rollouts and training loop
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env,
                callback=callback,
                rollout_buffers=[agent.rollout_buffer for agent in self.agents],
                n_rollout_steps=self.n_steps
            ) 
            if not continue_training:
                break
            iteration += 1
            
            # update agents
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            for agent in self.agents:
                agent._update_current_progress_remaining(self.num_timesteps//self.num_agents,
                                                         total_timesteps//self.num_agents)
            
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)
                
            self.train()

            callback.on_training_end()
        return self
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffers: List[RolloutBuffer],
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout) and reset rollout_buffers
        for i, agent in enumerate(self.agents):
            agent.policy.set_training_mode(False)
            rollout_buffers[i].reset()

        n_steps = 0
        if self.use_sde:
            for i, agent in enumerate(self.agents):
                agent.policy.reset_noise(env.num_envs)
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for i, agent in enumerate(self.agents):
                    agent.policy.reset_noise(env.num_envs)
                self.policy.reset_noise(env.num_envs)
                
            # predict actions
            all_clipped_actions, all_values, all_log_probs = self.feedforward(self._last_obs)
            actions = np.vstack(all_clipped_actions).transpose().reshape(-1) # reshape as (env, action)
            
            # env step
            new_obs, rewards, dones, infos = env.step(actions)
   
            
            self.num_timesteps += env.num_envs
            for agent in self.agents:
                 agent.num_timesteps += env.num_envs
            
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            n_steps += 1
            
             # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # TODO: make sure that the timeout is correctly handled
            # for idx, done in enumerate(dones):
            #     if (
            #         done
            #         and infos[idx].get("terminal_observation") is not None
            #         and infos[idx].get("TimeLimit.truncated", False)
            #     ):
            #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
            #         with th.no_grad():
            #             terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
            #         rewards[idx] += self.gamma * terminal_value
            for agent_id, rollout_buffer in enumerate(rollout_buffers):
                rollout_buffer.add(
                    self._last_obs[agent_id::self.num_agents],  # type: ignore[arg-type]
                    np.expand_dims(actions[agent_id::self.num_agents], -1),
                    rewards[agent_id::self.num_agents],
                    self._last_episode_starts[agent_id::self.num_agents],  # type: ignore[arg-type]
                    all_values[agent_id],
                    all_log_probs[agent_id]
                )
            self._last_obs = new_obs 
            self._last_episode_starts = dones
            
            self.update_agents_last_obs()
            self.update_last_episode_starts()
            
            
        _, values, _ = self.feedforward(self._last_obs)
        for i in range(self.num_agents):
            rollout_buffers[i].compute_returns_and_advantage(last_values=values[i], dones=dones[i::self.num_agents])

        # train reward_predictor

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True 
                 
    def feedforward(self, obs):
        all_actions = [None] * self.num_agents
        all_values = [None] * self.num_agents
        all_log_probs = [None] * self.num_agents
        all_clipped_actions = [None] * self.num_agents
        with th.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    obs_tensor = obs_as_tensor(obs[agent_id::self.num_agents], agent.policy.device)
                    all_actions[agent_id], all_values[agent_id], all_log_probs[agent_id] = agent.policy(obs_tensor)
                    all_values[agent_id] = all_values[agent_id].view(self.num_envs)
                    clipped_actions = all_actions[agent_id].cpu().numpy()
                    all_clipped_actions[agent_id] = clipped_actions
        return all_clipped_actions, all_values, all_log_probs
    
    def train(self) -> None:
        for agent in self.agents:
            agent.train()
    
         
    def update_agents_last_obs(self):
        for i in range(self.num_agents):
            last_obs = self._last_obs[i::self.num_agents]
            self.agents[i]._last_obs = last_obs
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        actions = np.zeros((self.num_envs*self.num_agents))
        for i, agent in enumerate(self.agents):
            ac, _ = agent.predict(observation[i::self.num_agents],
                                                        state,
                                                        episode_start,
                                                        deterministic)
            actions[i::self.num_agents] = ac
        return actions, None
    
    def update_last_episode_starts(self):
        for i in range(self.num_agents):
            last_episode_starts = self._last_episode_starts[i::self.num_agents]
            self.agents[i]._last_episode_starts = last_episode_starts