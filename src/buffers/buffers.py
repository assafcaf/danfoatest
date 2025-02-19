from collections import deque
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, NamedTuple
from enum import Enum
from gymnasium import spaces
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class PRMEpisodeData(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    experiment_rewards: np.ndarray
    true_rewards:np.ndarray
    aip:np.array
    fire_sucsses: np.array

    def __add__(self, other):
        """Dynamically concatenates PRMEpisodeData instances without relying on hardcoded field names."""
        combined_data = {}

        for field in self._fields:  # Iterate over all named fields
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            # If both values are not None, concatenate them
            if self_value is not None and other_value is not None:
                combined_data[field] = np.concatenate([self_value, other_value], axis=0)
            elif self_value is not None:
                combined_data[field] = self_value
            else:
                combined_data[field] = other_value

        return PRMEpisodeData(**combined_data)

    def __len__(self):
        return self.observations.shape[0]


class CRMEpisodeData(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    experiment_rewards: np.ndarray
    true_rewards:np.ndarray
    aip:np.array
    fire_sucsses: np.array

    def __add__(self, other):
        """Dynamically concatenates PRMEpisodeData instances without relying on hardcoded field names."""
        combined_data = {}

        for field in self._fields:  # Iterate over all named fields
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            # If both values are not None, concatenate them
            if self_value is not None and other_value is not None:
                combined_data[field] = np.concatenate([self_value, other_value], axis=0)
            elif self_value is not None:
                combined_data[field] = self_value
            else:
                combined_data[field] = other_value

        return PRMEpisodeData(**combined_data)

    def __len__(self):
        return self.states.shape[0]


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
        self.fire_sucsses = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

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
        self.fire_sucsses[self.pos] = np.array([info.get("fire_sucsses", False) for info in infos])
        
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
        fire_sucsses = np.zeros((batch_size, self.episode_length))
        aip = np.zeros((batch_size, self.episode_length))
        for i, (ep_idx, env_idx) in enumerate(zip(ep_indices, env_indices)):
            start_idx = self.episode_indices[ep_idx]
            end_idx = start_idx + self.episode_length
            observations[i] = self.observations[start_idx:end_idx, env_idx]
            actions[i] = self.actions[start_idx:end_idx, env_idx].squeeze()
            experiment_rewards[i] = self.experiment_rewards[start_idx:end_idx, env_idx]
            true_rewards[i] = self.true_rewards[start_idx:end_idx, env_idx]
            aip[i] = self.aip[start_idx:end_idx, env_idx]
            fire_sucsses[i] = self.fire_sucsses[start_idx:end_idx, env_idx]
        return PRMEpisodeData(observations=observations,
                              actions=actions,
                              experiment_rewards=experiment_rewards,
                              true_rewards=true_rewards,
                              aip=aip,
                              fire_sucsses=fire_sucsses)

class CRMShardReplayBuffer(PRMShardReplayBuffer):
    states: np.array
    def __init__(self,
                buffer_size,
                observation_space,
                action_space,
                device="auto",
                n_envs=1,
                optimize_memory_usage=False,
                episode_length=100, 
                state_space=None,
                n_agents=0, 
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
                        episode_length,
                        *args,
                        **kwargs)
        self.n_agents = n_agents
        a, b, c, d, e = self.observations.shape
        self.state_space = state_space
        self.states = np.zeros((a, b//n_agents, *state_space.shape), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        experiment_rewards: np.ndarray,
        states: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:  
        self.states[self.pos] = states
        super().add(obs, next_obs, action, reward, experiment_rewards, done, infos)

    def get_episodes(self, batch_size=1):
        """Retrieve full episodes from the buffer."""
        ##TODO: cheack if realy need to stroe experiment_rewards
        ep_indices = np.random.choice(len(self.episode_indices), size=batch_size, replace=True)
        env_batch_indices = np.random.choice(self.n_envs//self.n_agents, size=batch_size, replace=True)



        states = np.zeros((batch_size, self.episode_length, *self.state_space.shape))
        actions = np.zeros((batch_size,self.episode_length, self.n_agents))
        experiment_rewards =np.zeros((batch_size,self.episode_length, self.n_agents))
        true_rewards = np.zeros((batch_size,self.episode_length, self.n_agents))
        aip =  np.zeros((batch_size,self.episode_length, self.n_agents))

        for i, (ep_idx, env_idx) in enumerate(zip(ep_indices, env_batch_indices)):
            start_idx = self.episode_indices[ep_idx]
            end_idx = start_idx + self.episode_length
            start_env =  env_idx*self.n_agents
            env_env = start_env+ self.n_agents

            states[i] = self.states[start_idx:end_idx, env_idx]
            actions[i] = self.actions[start_idx:end_idx, start_env:env_env].squeeze()
            experiment_rewards[i] = self.experiment_rewards[start_idx:end_idx, start_env:env_env]
            true_rewards[i] = self.true_rewards[start_idx:end_idx, start_env:env_env]
            aip[i] = self.aip[start_idx:end_idx, start_env:env_env]
        return CRMEpisodeData(states, actions, experiment_rewards, true_rewards, aip)
    
    def sample(self, batch_size, env = None, agent_id=0):
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
                # Sample randomly the env idx
        # making sure that we sampling data for agent_id expirence only 
        env_indices = np.random.randint(0, high=self.n_envs//self.n_agents, size=(len(batch_inds),)) * self.n_agents + agent_id

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

