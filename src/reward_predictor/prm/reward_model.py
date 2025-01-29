import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .nn import MlpNetwork, CnnNetwork



class RewardModel:
    """
    Base class for reward models. Handles logging and basic structure.
    """
    def __init__(self, episode_logger):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1):
        pass

    def save_model_checkpoint(self):
        pass

    def load_model_checkpoint(self):
        pass


class ComparisonRewardPredictor(RewardModel):
    """
    Predicts reward values for trajectory segments using a neural network.
    """
    def __init__(
        self,
        num_envs,
        agent_logger,
        label_schedule,
        observation_space,
        action_space,
        epochs,
        device,
        lr=0.0001, 
        train_freq=1e4,
        comparison_collector_max_len=1000,
        save_every=200,
        id_=0,
        network_kwargs={'h_size': 64, 'emb_dim': 64}
    ):
        super().__init__(agent_logger)
        self.agent_logger = agent_logger
        self.label_schedule = label_schedule
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.train_freq = train_freq
        self.comparison_collector_max_len = comparison_collector_max_len
        self.id_ = id_
        self.save_every = save_every
        self.num_envs = num_envs
        self._initialize_buffers_and_model(observation_space, action_space, num_envs, network_kwargs)

    def _initialize_buffers_and_model(self, observation_space, action_space, num_envs, network_kwargs):
        self.recent_segments = deque(maxlen=200)

        self.model = CnnNetwork(
            observation_space=observation_space.shape, 
            n_actions=action_space.shape, 
            **network_kwargs
        ).to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def predict(self, obs, act):
        obs, act = obs.to(self.device).float(), act.to(self.device).long()
        return self.model(obs, act).cpu().detach().numpy()

    def train_predictor(self, verbose=False):
        """Train the reward predictor network."""
        self.comparison_collector.label_unlabeled_comparisons()
        losses = []

        for _ in range(self.epochs):
            batch = self.comparison_collector.sample_batch(min_batch_size=8)
            if not batch:
                continue

            left_obs, left_acts, right_obs, right_acts, labels = batch
            loss = self._train_step(left_obs, left_acts, right_obs, right_acts, labels)
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0
        if verbose:
            print(f"[Reward Predictor] Training completed. Avg loss: {avg_loss}")
        return avg_loss

    def _train_step(self, left_obs, left_acts, right_obs, right_acts, labels):
        left_obs, right_obs = left_obs.to(self.device).float(), right_obs.to(self.device).float()
        left_acts, right_acts = left_acts.to(self.device), right_acts.to(self.device)
        labels = labels.to(self.device)

        rewards_left = self.model(left_obs, left_acts).sum(dim=1, keepdim=True)
        rewards_right = self.model(right_obs, right_acts).sum(dim=1, keepdim=True)

        logits = torch.cat([rewards_left, rewards_right], dim=1)
        loss = self.loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model_checkpoint(self, path):
        torch.save(self.model.state_dict(), f"{path}/predictor_{self.id_}.pth")

    def load_model_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}/predictor_{self.id_}.pth", map_location=self.device))
