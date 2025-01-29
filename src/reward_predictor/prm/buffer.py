from collections import defaultdict
from typing import Generator

class PredictorBuffer:
    def __init__(self, num_envs: int):
        """
        Initializes a buffer to store trajectories for multiple environments.

        Args:
            num_envs (int): The number of environments.
        """
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        """
        Resets the buffer for all environments.
        """
        self.paths = [defaultdict(list) for _ in range(self.num_envs)]

    def store(self, obs, actions, pred_rewards, experiment_rewards, real_rewards, human_obs):
        """
        Stores step data for each environment.

        Args:
            obs: Observations for all environments.
            actions: Actions taken in all environments.
            pred_rewards: Predicted rewards for all environments.
            experiment_rewards: Experiment rewards for all environments.
            real_rewards: Real rewards for all environments.
            human_obs: Human observations for all environments.
        """
        for i in range(self.num_envs):
            self.paths[i]['obs'].append(obs[i])
            self.paths[i]['actions'].append(actions[i].item())
            self.paths[i]['rewards'].append(pred_rewards[i])
            self.paths[i]['experiment_rewards'].append(experiment_rewards[i])
            self.paths[i]['original_rewards'].append(real_rewards[i])

    def get(self) -> Generator[dict, None, None]:
        """
        Yields the collected trajectories and resets the buffer.

        Returns:
            Generator yielding paths for each environment.
        """
        paths = self.paths
        self.reset()
        yield from paths
