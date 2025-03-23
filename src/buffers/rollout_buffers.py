from .replay_buffers import PRMShardReplayBuffer
from stable_baselines3.common.buffers import RolloutBuffer

class PRMShardRolloutBuffer(RolloutBuffer):
    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 device = "auto",
                 gae_lambda = 1,
                 gamma = 0.99,
                 n_envs = 1,
                 replay_kwargs={}): # episode_length is needed from outside
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.replay_buffer = PRMShardReplayBuffer(observation_space=observation_space,
                                                  action_space=action_space,
                                                  device=device,
                                                  n_envs=n_envs,
                                                  **replay_kwargs)
    
    def replay_store(self, obs,next_obs, action, reward, experiment_rewards, done, infos):
        self.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,  # you might want to properly handle next_obs
            action=action,
            reward=reward,
            experiment_rewards=experiment_rewards,
            done=done,
            infos=infos,
        )
    
    def get_episodes(self, batch_size=1):
        return self.replay_buffer.get_episodes(batch_size=batch_size)