import random
from collections import deque


import numpy as np
import torch
import os
from .nn import MlpNetwork, CnnNetwork
from ..segment_sampling import sample_segment_from_path
from utils import corrcoef
from ..comparison_collectors import SyntheticComparisonCollector
from ..segment_sampling import segments_from_rand_rollout
from rl_agents.independent_dqn.buffer import PredictorBuffer


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

class RewardModel(object):
    def __init__(self, episode_logger):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def try_to_load_model_from_checkpoint(self):
        pass  # Nothing to load

class CrmComparisonRewardPredictor(RewardModel):
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""
    def __init__(self,
                 num_agents,
                 agent_loggers,
                 label_schedules,
                 fps,
                 observation_space,
                 action_space,
                 num_envs,
                 epochs,
                 network,
                 stacked_frames,
                 device,
                 lr=0.0001,
                 clip_length=0.1,
                 train_freq=1e4,
                 comparison_collector_max_len=1000,
                 weights_pth=None,
                 save_every=200,
                 pre_train=False):
        """ Initialize the reward predictor
        :param agent_logger: an AgentLogger object
        :param label_schedule: a LabelSchedule object
        :param fps: the frames per second of the environment
        :param observation_space: the observation space of the environment
        :param action_space: the action space of the environment
        :param stacked_frames: the number of frames to stack
        :param device: the device to run the model on
        :param lr: the learning rate for the model
        :param clip_length: the length of the video clip to use for training
        :param train_freq: how often to train the model
        :param comparison_collector_max_len: the maximum number of comparisons to store
        """
        self.agent_logger = agent_loggers[0]
        self.label_schedule = label_schedules[0]
        self.stacked_frames = stacked_frames
        self.device = device
        self.fps = fps
        self.epochs = epochs
        self.action_space = action_space
        self.num_envs = num_envs
        self.lr = lr
        self.clip_length = clip_length
        self.train_freq = train_freq
        self.comparison_collector_max_len = comparison_collector_max_len
        self.num_agents = num_agents
        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = clip_length * fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = train_freq  # How often should we train our predictor?
        self._elapsed_predictor_training_iters = 0

        # Build and initialize our comparison_collector
        self.comparison_collector = SyntheticComparisonCollector(max_len=comparison_collector_max_len)
        
        # Build and initialize our predictor model
        self.observation_space = observation_space
        
        self.discrete_action_space = hasattr(action_space, "shape")
        self.act_shape = (action_space.n,) if self.discrete_action_space else action_space.shape
        
        self.network = network
        self.model = self._build_model()
        if weights_pth:
            self.model.load_state_dict(torch.load(weights_pth))
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_op = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.ep_buffer = PredictorBuffer(num_envs)
        self.train_iter = 0
        self.save_every = save_every
        
    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + observation_space
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        obs_segments = obs_segments.to(self.device).float()
        act_segments = act_segments.to(self.device)
        return nn_predict_rewards(obs_segments, act_segments, network, self.observation_space, self.act_shape)

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up action placeholder
        if self.network == "cnn":
            # HACK Use a convolutional network for Atari
            # TODO Should check the input space dimensions, not the output space!
            net = CnnNetwork(observation_space=self.observation_space, h_size=64, emb_dim=3, n_actions=self.act_shape[0], num_outputs=self.num_agents)
             
        else:
            # In simple environments, default to a basic Multi-layer Perceptron (see TODO above)
            net = MlpNetwork(observation_space=self.observation_space, h_size=64, emb_dim=3, n_actions=self.act_shape[0], num_outputs=self.num_agents)


        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        return net.to(self.device)

    def store_step(self, obs, act, pred_rewards, real_rewards, human_obs):
        self.ep_buffer.store(obs, act, pred_rewards, real_rewards, human_obs)
    
    def get_paths(self):
        yield self.ep_buffer.get(), 0 # 0 is agent_id (to be compatible with PRM architecture)
    
    def predict(self, obs, act):
        """Predict the reward for 1 time step """
        r = self.model(obs.float(), act.long().to(self.device))
        return r.cpu().detach().numpy().squeeze()
        
    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        obs = path["obs"].to(self.device).float()
        action = path["actions"].to(self.device).long()

        return self.model(obs, action).detach().cpu()

    def path_callback(self, path, agent_id):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        # TODO: write a proper summary writer for torch
        self.agent_logger.log_episode(path)

        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        # print
        # if segment and random.random() < 0.25 or len(self.recent_segments) < 10:
            # self.recent_segments.append(segment)
            
        self.recent_segments.append(segment)
        
        # If we need more comparisons, then we build them from our recent segments
        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            self.comparison_collector.add_segment_pair(
                random.choice(self.recent_segments),
                random.choice(self.recent_segments))

        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            self.train_predictor()
            self.train_iter += 1
            if (self.train_iter % self.save_every) == 0:
                self.save_model_checkpoint()
            self._steps_since_last_training = 0

    def pre_trian(self, env_id, make_env, pretrain_labels, clip_length, n_steps, pretrain_iters, map_size, num_envs, same_color, gray_scale, same_dim):
        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=pretrain_labels * 2, same_color=same_color,
            clip_length_in_seconds=clip_length, workers=num_envs, gray_scale=gray_scale, same_dim=same_dim,
            stacked_frames=self.stacked_frames, max_episode_steps=n_steps, map_size=map_size)
        for i in range(pretrain_labels):  # Turn our random segments into comparisons
            self.comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])

        # Sleep until the human has labeled most of the pretraining comparisons
        while len(self.comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            self.comparison_collector.label_unlabeled_comparisons()
            print("%s synthetic labels generated... " % (len(self.comparison_collector.labeled_comparisons)))

        # Start the actual training
        losses = []
        for i in range(pretrain_iters):
            loss = self.train_predictor()  # Train on pretraining labels
            losses.append(loss)
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... (Err: %s)" % (i, pretrain_iters, np.mean(losses)))
                
    def train_predictor(self, verbose=False, pre_train=False):
        epches = 1 if pre_train else self.epochs
        losses = np.zeros(epches)
        self.comparison_collector.label_unlabeled_comparisons()
        for i in range(epches):
            

            minibatch_size = min(8, len(self.comparison_collector.labeled_decisive_comparisons))
            labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
            left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
            left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
            right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
            right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
            labels = np.asarray([comp['label'] for comp in labeled_comparisons])
            losses[i] = self._train_step(left_obs, left_acts, right_obs, right_acts, labels).item()
        self._elapsed_predictor_training_iters += 1
        loss = np.mean(losses)
        
        if not pre_train:
            self._write_training_summaries(loss)
        
        if verbose:
            print("Reward predictor training iter %s (Err: %s)" % (self._elapsed_predictor_training_iters, loss))
            
        return loss
    
    def _train_step(self, left_obs, left_acts, right_obs, right_acts, labels):
        """ Train the model on a single batch """
        
        # move to torch
        left_obs = torch.tensor(np.stack(left_obs))
        left_acts = torch.tensor(left_acts)
                
        right_obs = torch.tensor(np.stack(right_obs))
        right_acts = torch.tensor(right_acts)
        
        labels = torch.tensor(labels).to(self.device)
        
        # predict rewards
        rewards_left = self._predict_rewards(left_obs, left_acts, self.model)
        rewards_right = self._predict_rewards(right_obs, right_acts, self.model)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = rewards_left.sum(axis=1, keepdim=True)
        segment_reward_pred_right = rewards_right.sum(axis=1, keepdim=True)
        reward_logits = torch.cat([segment_reward_pred_left, segment_reward_pred_right], axis=1) # (batch_size, 2)

        loss = self.loss(reward_logits, labels)
        
        self.train_op.zero_grad()
        loss.backward()
        self.train_op.step()
        torch.cuda.empty_cache()
        return loss
   
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

    def copy(self):
        new_predictor = CrmComparisonRewardPredictor(
            agent_logger=self.agent_logger,
            label_schedule=self.label_schedule,
            fps=self.fps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            num_envs=self.num_envs,
            epochs=self.epochs,
            stacked_frames=self.stacked_frames,
            lr=self.lr,
            device=self.device,
            clip_length=self.clip_length,
            train_freq=self.train_freq,
            comparison_collector_max_len=self.comparison_collector_max_len,
            network=self.network)
        new_predictor.model.load_state_dict(self.model.state_dict())

        return new_predictor
    
    def dump(self, step):
        self.agent_logger.dump(step)
   
    def save_model_checkpoint(self):
        f_name = os.path.join(self.agent_logger.summary_writer.dir, "checkpoints", "predictor.pth")
        torch.save(self.model.state_dict(), f_name)