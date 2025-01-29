import threading
import time
import torch.multiprocessing as mp

class RLWithRewardPredictor:
    """
    Wrapper class that integrates a reinforcement learning agent (DQNRP) and a reward prediction network,
    handling asynchronous reward predictor training with multi-agent parallelization.
    """
    def __init__(self, rl_agent, reward_predictor, train_rp_freq=1000, async_rp_training=True, parallel_agents=True):
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
        self.async_rp_training = async_rp_training
        self.parallel_agents = parallel_agents
        self.total_steps = 0
        self.training_thread = None

        if self.parallel_agents:
            self.pool = mp.Pool(processes=mp.cpu_count())

    def train(self, total_timesteps):
        """
        Train the RL agent and periodically train the reward predictor.

        :param total_timesteps: Total number of timesteps to train the RL agent.
        """
        while self.total_steps < total_timesteps:
            # Collect rollouts with the RL agent
            rollouts = self.rl_agent.collect_rollouts(
                env=self.rl_agent.env,
                callback=self.rl_agent._callback,
                train_freq=self.rl_agent.train_freq,
                replay_buffer=self.rl_agent.replay_buffer,
                action_noise=self.rl_agent.action_noise,
                learning_starts=self.rl_agent.learning_starts,
                log_interval=self.rl_agent.log_interval,
            )

            self.total_steps += rollouts.timesteps

            # Train the reward predictor periodically
            if self.total_steps % self.train_rp_freq == 0:
                if self.async_rp_training:
                    self._train_reward_predictor_async()
                else:
                    self.train_reward_predictor()

            # Train the RL agent
            self.rl_agent.train_step()

    def _train_reward_predictor_async(self):
        """
        Train the reward predictor asynchronously in a separate thread.
        """
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self.train_reward_predictor, daemon=True)
            self.training_thread.start()

    def train_reward_predictor(self):
        """
        Train the reward predictor network, with parallelization if enabled.
        """
        if self.parallel_agents:
            avg_loss = sum(self.pool.map(self._train_single_predictor, self.reward_predictor.predictors)) / len(self.reward_predictor.predictors)
        else:
            avg_loss = self.reward_predictor.train_predictor(verbose=True)
        print(f"[Reward Predictor] Average training loss: {avg_loss}")

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
