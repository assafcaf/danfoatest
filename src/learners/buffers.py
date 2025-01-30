from collections import deque
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, NamedTuple
from enum import Enum
from gymnasium import spaces

class EpisodeData(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    true_rewards: np.ndarray

class RLWithRewardPredictorBuffer(ReplayBuffer):
    experiment_rewards: np.array
    def __init__(self,
                buffer_size,
                observation_space,
                action_space,
                device="auto",
                n_envs=1,
                optimize_memory_usage=False,
                episode_length=100, 
                *args,
                **kwargs):
        assert buffer_size%episode_length == 0, (f"Error: Buffer size ({buffer_size}) must be a multiple of episode length ({episode_length}) "
                  f"to ensure complete episode storage and retrieval.")
        super().__init__(buffer_size,
                        observation_space,
                        action_space,
                        device,
                        n_envs,
                        optimize_memory_usage,
                        *args,
                        **kwargs)
        self.episode_length = episode_length
        self.episode_indices = []  # Track episode start indices
        self.true_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
   
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a transition and track episode indices."""
        if len(self.episode_indices) == 0 or self.episode_indices[-1] + self.episode_length <= self.pos:
            self.episode_indices.append(self.pos)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.true_rewards[self.pos] = np.array([info.get("true_reward", False) for info in infos])
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        
    def get_episodes(self, n_episodes=1):
        """Retrieve full episodes from the buffer."""
        ##TODO: cheack if realy need to stroe experiment_rewards
        sampled_indices = np.random.choice(len(self.episode_indices), size=n_episodes, replace=False)
        observations = np.zeros((self.n_envs*n_episodes, self.episode_length, *self.observation_space.shape))
        actions = np.zeros((self.n_envs*n_episodes, self.episode_length))
        rewards = np.zeros((self.n_envs*n_episodes, self.episode_length))
        true_rewards = np.zeros((self.n_envs*n_episodes, self.episode_length))
        env_cnt = 0
        for idx in sampled_indices:
            start_idx = self.episode_indices[idx]
            end_idx = start_idx + self.episode_length
            observations[env_cnt:env_cnt+self.n_envs] = np.swapaxes(self.observations[start_idx:end_idx], 0, 1)
            actions[env_cnt:env_cnt+self.n_envs] = np.swapaxes(self.actions[start_idx:end_idx], 0, 1).squeeze()
            rewards[env_cnt:env_cnt+self.n_envs] = np.swapaxes(self.rewards[start_idx:end_idx], 0, 1)
            true_rewards[env_cnt:env_cnt+self.n_envs] = np.swapaxes(self.true_rewards[start_idx:end_idx], 0, 1)

            env_cnt += self.n_envs
        return EpisodeData(observations, actions, rewards, true_rewards)
