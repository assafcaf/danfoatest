from collections import deque
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class RLWithRewardPredictorBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1, episode_length=100):
        adjusted_buffer_size = (buffer_size // episode_length) * episode_length
        assert adjusted_buffer_size != buffer_size, (
        f"Warning: Buffer size ({buffer_size}) must be a multiple of episode length ({episode_length}) "
        f"to ensure complete episode storage and retrieval. Adjusting to {adjusted_buffer_size}."
    )
        super().__init__(adjusted_buffer_size, observation_space, action_space, device, n_envs)
        self.episode_length = episode_length
        self.episode_indices = []  # Track episode start indices
        
    def add(self, obs, action, reward, next_obs, done):
        """Add a transition and track episode indices."""
        if len(self.episode_indices) == 0 or self.episode_indices[-1] + self.episode_length <= self.pos:
            self.episode_indices.append(self.pos)
        super().add(obs, action, reward, next_obs, done)
        
    def get_episodes(self, n_episodes=1):
        """Retrieve full episodes from the buffer."""
        sampled_indices = np.random.choice(len(self.episode_indices), size=n_episodes, replace=False)
        episodes = []
        for idx in sampled_indices:
            start_idx = self.episode_indices[idx]
            end_idx = start_idx + self.episode_length
            episodes.append({
                "observations": self.observations[start_idx:end_idx],
                "actions": self.actions[start_idx:end_idx],
                "rewards": self.rewards[start_idx:end_idx],
                "next_observations": self.next_observations[start_idx:end_idx],
                "dones": self.dones[start_idx:end_idx],
            })
        return episodes
