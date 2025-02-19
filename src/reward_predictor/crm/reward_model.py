import random
from collections import deque
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
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""
    def __init__(
        self,
        num_outputs,
        agent_logger,
        observation_space,
        action_space,
        epochs,
        device,
        lr=0.0001, 
        save_every=200,
        id_=0,
        network_kwargs={'h_size': 64, 'emb_dim': 8}
    ):
        super().__init__(agent_logger)
        self.agent_logger = agent_logger
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.id_ = id_
        self.save_every = save_every
        self.observation_space = observation_space
        network_kwargs['num_outputs'] = num_outputs
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
    
    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        obs = path["obs"].to(self.device).float()
        action = path["actions"].to(self.device).long()

        return self.model(obs, action).detach().cpu()

    def transform_batch(self, batch):
        """
        obs.shape-> envs*ep_length*obs_shape"""
        l = batch.states.shape[0]//2

        # left segments
        left_obs = torch.tensor(batch.states[:l]).to(self.device).float()
        left_acts = torch.tensor(batch.actions[:l]).to(self.device)
        left_r = batch.experiment_rewards[:l].sum(axis=2).sum(axis=1)
        
        # rights segments
        right_obs = torch.tensor(batch.states[l:]).to(self.device).float()
        right_acts = torch.tensor(batch.actions[l:]).to(self.device)
        right_r = batch.experiment_rewards[l:].sum(axis=2).sum(axis=1)

        # lables
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

    def _train_step(self, left_obs, left_acts, right_obs, right_acts, labels):
        """ Train the model on a single batch """
        
        rewards_left = self.predict_batch(left_obs, left_acts)
        rewards_right = self.predict_batch(right_obs, right_acts)

        logits = torch.stack([rewards_left.mean(axis=2).sum(1), rewards_right.mean(axis=2).sum(1)], dim=1)
        loss = self.loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        predicted_rewards = torch.cat([rewards_left.flatten(), rewards_right.flatten()], dim=0).detach().cpu().numpy()
        return loss, predicted_rewards.squeeze()
   
    def predict_batch(self, obs_segments, act_segments):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + observation_space
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :return: tensor with shape = (batch_size, segment_length)
        """
        # TODO: make this work with pytorch
        
        batchsize = obs_segments.shape[0]
        segment_length = obs_segments.shape[1]

        # Temporarily chop up segments into individual observations and actions
        # TODO: makesure its works fine without transpose (observation_space)
        obs = obs_segments.view((-1,) + self.observation_space.shape)
        acts = act_segments.view((-1, act_segments.shape[-1]))

        # # Run them through our neural network
        rewards = self.model(obs, acts)

        # # Group the rewards back into their segments
        # return tf.reshape(rewards, (batchsize, segment_length))
        return rewards.view((batchsize, segment_length, -1))
   
    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("predictor/_loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            idx = random.sample(range(len(recent_paths)), min(len(recent_paths), 10))
            validation_obs = np.asarray([path["obs"] for i, path in enumerate(recent_paths) if i in idx])
            validation_acts = np.asarray([path["actions"] for i, path in enumerate(recent_paths) if i in idx])
            q_value = self._predict_rewards(torch.from_numpy(validation_obs),
                                            torch.from_numpy(validation_acts),
                                            self.model).detach().cpu().numpy()
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for i, path in enumerate(recent_paths) if i in idx])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))
            torch.cuda.empty_cache()
            
        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))
    
    def buffer_usage(self):
        return self.comparison_collector.buffer_usage()

    def save_model_checkpoint(self, path):
        torch.save(self.model.state_dict(), f"{path}/predictor_{self.id_}.pth")

    def log_step(self, batch, predicted_rewards, loss):
        # predicted rewards for eaten and uneaten apples

        rewards = batch.true_rewards.reshape(-1)
        actions = batch.actions.reshape(-1)
        aip = batch.aip.reshape(-1)
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
            'on_action/stay':np.nanmean(predicted_rewards[actions==4]),
            'on_action/turn': np.nanmean(predicted_rewards[(actions == 5) | (actions == 6)]),
            'on_action/fire': np.nanmean(predicted_rewards[actions==7]),
            

            'predicter rewards by appeals in porximity/0': np.nanmean(predicted_rewards[(rewards==1) & (aip==0)]),
            'predicter rewards by appeals in porximity/1': np.nanmean(predicted_rewards[(rewards==1) & (aip==1)]),
            'predicter rewards by appeals in porximity/2': np.nanmean(predicted_rewards[(rewards==1) & (aip==2)]),
            'predicter rewards by appeals in porximity/3+': np.nanmean(predicted_rewards[(rewards==1) & (aip>=3)]),

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
