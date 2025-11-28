import threading
import time
import torch.multiprocessing as mp

class IndependentRLRPLearner:
    """
    Wrapper class that integrates a reinforcement learning agent (DQNRP) and a reward prediction network,
    handling asynchronous reward predictor training with multi-agent parallelization.
    """
    def __init__(self,
                 rl_agent,
                 reward_predictor,
                 train_rp_freq=1000,
                 rp_learning_starts=10,
                 batch_size=4):
        """
        Initializes the wrapper.

        :param rl_agent: The reinforcement learning agent (DQNRP).
        :param reward_predictor: The reward prediction network (PrmComparisonRewardPredictor).
        :param train_rp_freq: Frequency (in steps) at which to train the reward predictor.
        :param async_rp_training: Whether to train the reward predictor asynchronously.
        :param parallel_agents: Whether to parallelize multi-agent reward prediction.
        """
        self.rl_agent = rl_agent
        self.reward_predictor = reward_predictor
        self.train_rp_freq = train_rp_freq
        self.batch_size = batch_size
        self.total_steps = 0
        self.total_episodes = 0
        self.training_thread = None
        self.rp_learning_starts = rp_learning_starts

    def learn(self,
              learner,
              total_timesteps,
              callback,
              log_interval,
              tb_log_name="run",
              reset_num_timesteps=True,
              progress_bar=False,
        ):
        if learner == "dqn":
            learn_fn =  self.learn_dqn
        elif learner == "ppo":
            learn_fn = self.learn_ppo

        return learn_fn(total_timesteps,
                        callback,
                        log_interval,
                        tb_log_name,
                        reset_num_timesteps,
                        progress_bar)   

    def learn_dqn(self,
            total_timesteps,
            callback,
            log_interval,
            tb_log_name="run",
            reset_num_timesteps=True,
            progress_bar=False):
        """
        Train the RL agent and periodically train the reward predictor.

        :param total_timesteps: Total number of timesteps to train the RL agent.
        """
        total_timesteps, callback = self.rl_agent._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
        ep_cnt = 0
        self.rl_agent.update_agents_last_obs()
        for agent, logger in zip(self.rl_agent.agents, self.loggers):
            agent.set_logger(logger)

        callback.on_training_start(locals(), globals())
        assert self.rl_agent.env is not None, "You must set the environment before calling learn()"

        while self.total_steps < total_timesteps:
            # Collect rollouts with the RL agent
            rollout = self.rl_agent.collect_rollouts(
                env=self.rl_agent.env,
                callback=callback,
                train_freq=self.rl_agent.train_freq,
                replay_buffer=self.rl_agent.replay_buffer,
                action_noise=self.rl_agent.action_noise,
                learning_starts=self.rl_agent.learning_starts,
                log_interval=log_interval,
            )
            if not rollout.continue_training:
                break
            self.total_steps += rollout.episode_timesteps
            self.total_episodes += rollout.n_episodes//self.rl_agent.env.num_envs
            # Train the reward predictor periodically
            if self.total_episodes > self.rp_learning_starts and self.total_episodes > ep_cnt:
                batch = self.rl_agent.replay_buffer.get_episodes(batch_size=self.batch_size)
                rp_log_dict = self.train_reward_predictor(batch=batch)
                for k, v in rp_log_dict.items():
                    self.rl_agent.logger.record(k, v)
            # Train the RL agent
            gradient_steps = self.rl_agent.gradient_steps if self.rl_agent.gradient_steps >= 0 else rollout.episode_timesteps
            ep_cnt = self.total_episodes
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.rl_agent.train_multiagent(batch_size=self.rl_agent.batch_size, gradient_steps=gradient_steps)
        callback.on_training_end()
        return self
    
    def learn_ppo(self,
            total_timesteps,
            callback,
            log_interval,
            tb_log_name="run",
            reset_num_timesteps=True,
            progress_bar=False):
        """
        Train the RL agent and periodically train the reward predictor.

        :param total_timesteps: Total number of timesteps to train the RL agent.
        """
        iteration = 0
        total_timesteps, callback = self.rl_agent._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
        callback.on_training_start(locals(), globals())
        assert self.rl_agent.env is not None, "You must set the environment before calling learn()"

        while self.total_steps < total_timesteps:
            # Collect rollouts with the RL agent
            continue_training = self.rl_agent.collect_rollouts(self.rl_agent.env, callback, self.rl_agent.rollout_buffer, n_rollout_steps=self.rl_agent.n_steps)

            if not continue_training:
                break
            iteration += 1
            self.rl_agent._update_current_progress_remaining(self.rl_agent.num_timesteps, total_timesteps)
            # Train the reward predictor periodically

            rp_log_dict = self.train_reward_predictor()
            for k, v in rp_log_dict.items():
                self.rl_agent.logger.record(k, v)

            # Train the RL agent
          # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self.rl_agent._dump_logs(iteration)
            self.rl_agent.train()
        return self

    def train_reward_predictor(self, batch):
        """
        Train the reward predictor network, with parallelization if enabled.
        """
        rp_log_dict = self.reward_predictor.train_predictor(batch=batch, verbose=False)
        return rp_log_dict

    def _train_single_predictor(self, predictor):
        """
        Helper function for training a single reward predictor in parallel.
        """
        return predictor.train_predictor(verbose=False)

    def save(self, path):
        """
        Save the RL agent and reward predictor.

        :param path: Directory path to save the models.
        """
        rl_path = f"{path}/rl_agent"
        rp_path = f"{path}/reward_predictor"

        self.rl_agent.save(rl_path)
        self.reward_predictor.save_model_checkpoint(rp_path)
        print(f"[Save] Models saved to {path}")

    def load(self, path):
        """
        Load the RL agent and reward predictor.

        :param path: Directory path to load the models from.
        """
        rl_path = f"{path}/rl_agent"
        rp_path = f"{path}/reward_predictor"

        self.rl_agent = self.rl_agent.load(rl_path)
        self.reward_predictor.load_model_checkpoint(rp_path)
        print(f"[Load] Models loaded from {path}")

    def evaluate(self, env, num_episodes=10):
        """
        Evaluate the RL agent in the environment.

        :param env: The evaluation environment.
        :param num_episodes: Number of episodes to run evaluation.
        """
        rewards = []

        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

        avg_reward = sum(rewards) / num_episodes
        print(f"[Evaluation] Average Reward over {num_episodes} episodes: {avg_reward}")
        return avg_reward
