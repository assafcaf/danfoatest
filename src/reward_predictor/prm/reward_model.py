import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .nn import MlpNetwork, CnnNetwork
from ..utils import corrcoef


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
        agent_logger,
        observation_space,
        action_space,
        epochs,
        device,
        lr=0.0001, 
        save_every=200,
        id_=0,
        network_kwargs={'h_size': 64, 'emb_dim': 64}
    ):
        super().__init__(agent_logger)
        self.agent_logger = agent_logger
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.id_ = id_
        self.save_every = save_every
        self.observation_space = observation_space
        self._initialize_buffers_and_model(observation_space, action_space, network_kwargs)

    def _initialize_buffers_and_model(self, observation_space, action_space, network_kwargs):
        self.recent_segments = deque(maxlen=200)

        self.model = CnnNetwork(
            observation_space=observation_space, 
            n_actions=action_space, 
            **network_kwargs
        ).to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def predict(self, obs, act):
        obs, act = obs.to(self.device).float(), act.to(self.device).long()
        return self.model(obs, act).cpu().detach().numpy()

    def transform_batch(self, batch):
        """
        obs.shape-> envs*ep_length*obs_shape"""
        l = batch.observations.shape[0]//2

        # left
        left_obs = torch.tensor(batch.observations[:l]).to(self.device).float()
        left_acts = torch.tensor(batch.actions[:l]).to(self.device)

        # rights
        right_obs = torch.tensor(batch.observations[l:]).to(self.device).float()
        right_acts = torch.tensor(batch.actions[l:]).to(self.device)

        # lable
        left_r = batch.true_rewards[:l].sum(axis=0)
        right_r = batch.true_rewards[l:].sum(axis=0)
        labels =  torch.tensor(left_r < right_r).to(self.device).long()

        return left_obs, left_acts, right_obs, right_acts, labels
    
    def train_predictor(self, batch, verbose=False):
        """Train the reward predictor network."""
        losses = []
        left_obs, left_acts, right_obs, right_acts, labels = self.transform_batch(batch)
        for _ in range(self.epochs):
            loss, predicted_rewards = self._train_step(left_obs, left_acts, right_obs, right_acts, labels)
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0
        if verbose:
            print(f"[Reward Predictor] Training completed. Avg loss: {avg_loss}")
        
        log_dict = self.log_step(batch, predicted_rewards, avg_loss)
        torch.cuda.empty_cache()
        return log_dict

    def _predict_rewards(self, obs_segments, act_segments):
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
        obs = obs_segments.view((-1,) + self.observation_space.shape)
        acts = act_segments.view((-1, 1))

        # # Run them through our neural network
        rewards = self.model(obs, acts)

        # # Group the rewards back into their segments
        # return tf.reshape(rewards, (batchsize, segment_length))
        return rewards.view((batchsize, segment_length))
   
    def _train_step(self, left_obs, left_acts, right_obs, right_acts, labels):

        rewards_left = self._predict_rewards(left_obs, left_acts)
        rewards_right = self._predict_rewards(right_obs, right_acts)

        logits = torch.cat([rewards_left.sum(dim=0, keepdim=True), rewards_right.sum(dim=0, keepdim=True)], dim=0).transpose(1, 0)
        loss = self.loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        predicted_rewards = torch.cat([rewards_left.flatten(), rewards_right.flatten()], dim=0).detach().cpu().numpy()
        return loss, predicted_rewards.squeeze()

    def save_model_checkpoint(self, path):
        torch.save(self.model.state_dict(), f"{path}/predictor_{self.id_}.pth")

    def log_step(self, batch, predicted_rewards, loss):
        # predicted rewards for eaten and uneaten apples

        a, b = batch.rewards.shape
        rewards = batch.true_rewards.reshape(a * b)
        actions = batch.actions.reshape(a * b)

        positive_reward_mean = np.nanmean(predicted_rewards[rewards==1])
        positive_reward_std = np.nanstd(predicted_rewards[rewards==1])

        zero_reward_mean = np.nanmean(predicted_rewards[rewards==0])
        zero_reward_std = np.nanstd(predicted_rewards[rewards==0])
        correlations = corrcoef(rewards, predicted_rewards)


        # predicted rewards per action
        action_reward_map = {
            'predictor/_loss': loss,
            "predictor/correlations": correlations,

            'on_action/move left': np.nanmean(predicted_rewards[actions==0]), 
            'on_action/move right':np.nanmean(predicted_rewards[actions==1]),
            'on_action/move up': np.nanmean(predicted_rewards[actions==2]),
            'on_action/move down':np.nanmean(predicted_rewards[actions==3]),
            'on_action/turn': np.nanmean(predicted_rewards[(actions == 5) | (actions == 6)])
,
            'on_action/fire': np.nanmean(predicted_rewards[actions==7]),

            # on apples eaten or not
            "outcome_avg/apple_eaten": positive_reward_mean,
            "outcome_avg/no_apple_eaten": zero_reward_mean,
            "outcome_avg/delta": positive_reward_mean-zero_reward_mean,

            "outcome_std/apple_eaten": positive_reward_std,
            "outcome_std/no_apple_eaten": zero_reward_std,
            "outcome_std/delta": positive_reward_std-zero_reward_std,
        }

        return action_reward_map
        

    def load_model_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}/predictor_{self.id_}.pth", map_location=self.device))
