import random
from collections import deque


import numpy as np
import torch
import os
from .nn import MlpNetwork, CnnNetwork
from ..comparison_collectors import SyntheticComparisonCollector
from .buffer import PredictorBuffer


def nn_predict_rewards(obs_segments, act_segments, network, observation_space, act_shape):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + observation_space
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :param observation_space: a tuple representing the shape of the observation space
    :param act_shape: a tuple representing the shape of the action space
    :return: tensor with shape = (batch_size, segment_length)
    """
    # TODO: make this work with pytorch
    
    batchsize = (obs_segments).shape[0]
    segment_length = (obs_segments).shape[1]

    # Temporarily chop up segments into individual observations and actions
    # TODO: makesure its works fine without transpose (observation_space)
    obs = obs_segments.view((-1,) + observation_space.shape)
    acts = act_segments.view((-1, 1))

    # # Run them through our neural network
    rewards = network(obs, acts)

    # # Group the rewards back into their segments
    # return tf.reshape(rewards, (batchsize, segment_length))
    return rewards.view((batchsize, segment_length))

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
        network_kwargs={'h_size': 64,
                         'emb_dim': 64}
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
        self.comparison_collector = SyntheticComparisonCollector(max_len=self.comparison_collector_max_len)

        self.model = CnnNetwork(observation_space=observation_space.shape, n_actions=action_space.shape, **network_kwargs).to(self.device)

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.ep_buffer = PredictorBuffer(num_envs)

    def predict(self, obs, act):
        obs, act = obs.to(self.device).float(), act.to(self.device).long()
        return self.model(obs, act).cpu().detach().numpy()

    def train_predictor(self, verbose=False):
        self.comparison_collector.label_unlabeled_comparisons()
        losses = []

        for _ in range(self.epochs):
            batch = self.comparison_collector.sample_batch(min_batch_size=8)
            if not batch:
                continue

            left_obs, left_acts, right_obs, right_acts, labels = batch
            loss = self._train_step(left_obs, left_acts, right_obs, right_acts, labels)
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        if verbose:
            print(f"Training completed. Avg loss: {avg_loss}")
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


class PrmComparisonRewardPredictor(RewardModel):
    """
    Manages multiple independent ComparisonRewardPredictor instances for MARL.
    """
    def __init__(self, num_agents, agent_loggers, label_schedules, **kwargs):
        self.num_agents = num_agents
        self.predictors = [
            ComparisonRewardPredictor(agent_logger=agent_loggers[i], label_schedule=label_schedules[i], **kwargs)
            for i in range(num_agents)
        ]

    def predict(self, obs, act):
        results = []
        for i, predictor in enumerate(self.predictors):
            predictions = predictor.predict(obs[i::self.num_agents], act[i::self.num_agents])
            results.append(predictions)
        return np.concatenate(results)

    def train_predictor(self, verbose=False):
        return np.mean([predictor.train_predictor(verbose) for predictor in self.predictors])

    def save_model_checkpoint(self):
        for predictor in self.predictors:
            predictor.save_model_checkpoint()

    def load_model_checkpoint(self, path):
        for predictor in self.predictors:
            predictor.load_model_checkpoint(path)
