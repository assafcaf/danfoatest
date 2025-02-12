from collections import deque
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, NamedTuple
from enum import Enum
from gymnasium import spaces

class EpisodeData(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    experiment_rewards: np.ndarray
    true_rewards:np.ndarray
    aip:np.array

class PRMShardReplayBuffer(ReplayBuffer):
    experiment_rewards: np.array
    aip: np.array
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
        self.experiment_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.true_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.aip = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    @property
    def predicted_rewards(self):
        return self.rewards
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        experiment_rewards: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:

        
        if len(self.episode_indices) == 0 or self.episode_indices[-1] + self.episode_length <= self.pos:
            self.episode_indices.append(self.pos)
          # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
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
        self.predicted_rewards[self.pos] = np.array(reward)
        self.experiment_rewards[self.pos] = np.array(experiment_rewards)
        self.true_rewards[self.pos] = np.array([info.get("true_reward", False) for info in infos])
        self.dones[self.pos] = np.array(done)
        self.aip[self.pos] = np.array([info.get("aip", False) for info in infos])
        
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_episodes(self, batch_size=1):
        """Retrieve full episodes from the buffer."""
        ##TODO: cheack if realy need to stroe experiment_rewards
        ep_indices = np.random.choice(len(self.episode_indices), size=batch_size, replace=True)
        env_indices = np.random.choice(self.n_envs, size=batch_size, replace=True)
        observations = np.zeros((batch_size, self.episode_length, *self.observation_space.shape))
        actions = np.zeros((batch_size, self.episode_length))
        experiment_rewards = np.zeros((batch_size, self.episode_length))
        true_rewards = np.zeros((batch_size, self.episode_length))

        aip = np.zeros((batch_size, self.episode_length))
        for i, (ep_idx, env_idx) in enumerate(zip(ep_indices, env_indices)):
            start_idx = self.episode_indices[ep_idx]
            end_idx = start_idx + self.episode_length
            observations[i] = self.observations[start_idx:end_idx, env_idx]
            actions[i] = self.actions[start_idx:end_idx, env_idx].squeeze()
            experiment_rewards[i] = self.experiment_rewards[start_idx:end_idx, env_idx]
            true_rewards[i] = self.true_rewards[start_idx:end_idx, env_idx]
            aip[i] = self.aip[start_idx:end_idx, env_idx]
        return EpisodeData(observations, actions, experiment_rewards, true_rewards, aip)

