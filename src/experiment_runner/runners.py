
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# local imports
from callbacks import SingleAgentCallback
from configs import Config
from reward_predictor import AgentLoggerSb3, LabelAnnealer, function_wrapper, PrmComparisonRewardPredictor
from learners.rl_predictor_warpper import RLWithRewardPredictor, RLWithRewardPredictorBuffer
from env import parallel_env
from rl_agents import DQN, IndependentDQN, PPO, CnnFeatureExtractor

import os
import yaml
import torch
import supersuit as ss
from datetime import datetime
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecTransposeImage, VecMonitor





class BaseRunner:
    OUTPUT_CHANNELS = ["stdout", "tensorboard", 'csv'] 
    def __init__(self, config_path):
        # Load configuration
        config_dict = self.load_yaml(config_path)
        self.config = Config(config_dict)

        # Set CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.EXPERIMENT_PARAMETERS.gpu_id)
        self.learner_kwargs = {
            "learning_rate": self.config.RL_PARAMETERS.learning_rate,
            "batch_size": self.config.RL_PARAMETERS.batch_size,
            "tau" :self.config.RL_PARAMETERS.tau,
            "gamma": self.config.RL_PARAMETERS.gamma,
            "train_freq": self.config.RL_PARAMETERS.train_freq,
            "exploration_fraction": self.config.RL_PARAMETERS.exploration_fraction,
            "learning_starts": int(self.config.RL_PARAMETERS.learning_starts),
            "buffer_size": int(self.config.RL_PARAMETERS.buffer_size),
            "exploration_final_eps": self.config.RL_PARAMETERS.exploration_final_eps,
        }
   
    def init_experiment(self, agent_fn):
        dir_name = self.config.EXPERIMENT_PARAMETERS.run_name_template.format(
            experiment=self.experiment,
            metric=self.config.EXPERIMENT_PARAMETERS.metric,
            spawn_speed=self.config.RL_PARAMETERS.spawn_speed
        )
        self.config.EXPERIMENT_PARAMETERS.run_name_template = dir_name
        log_dir = os.path.join(self.log_dir, dir_name, "run_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        vec_env = self.setup_environment(self.config, self.config.ENV_PARAMETERS.num_envs)
        eval_env = self.setup_environment(self.config, 1)
        agent = self.setup_agent(vec_env, agent_fn, log_dir)

        render_frequency = 2000 if self.config.RL_PARAMETERS.learner == "dqn" else 20

        if self.config.RL_PARAMETERS.learner == "ppo":
            log_interval = 1
        else:
            log_interval = self.config.ENV_PARAMETERS.num_agent * self.config.ENV_PARAMETERS.num_envs

        callback = SingleAgentCallback(
            eval_env,
            verbose=0,
            render_frequency=render_frequency,
            deterministic=False,
            args=dict(self.config),
            learner=self.config.RL_PARAMETERS.learner,
        )
        return agent, callback, log_interval

    def setup_environment(self, config, num_envs):
        """Configures the environment."""
        env = parallel_env(
            num_agents=config.ENV_PARAMETERS.num_agent,
            ep_length=config.ENV_PARAMETERS.ep_length,
            penalty=config.EXPERIMENT_PARAMETERS.penalty,
            spawn_speed=config.RL_PARAMETERS.spawn_speed,
            metric=config.EXPERIMENT_PARAMETERS.metric
        )
        env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
        env = ss.frame_stack_v1(env, self.config.ENV_PARAMETERS.num_frames)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env,
            num_vec_envs=num_envs,
            num_cpus=config.ENV_PARAMETERS.num_cpus,
            base_class="stable_baselines3"
        )
        env = VecMonitor(env)
        env = VecTransposeImage(env)
        return env

    def setup_agent(self, vec_env, agent_fn, log_dir):
        loggers = self.setup_loggers(log_dir)

        policy_kwargs = dict(
            features_extractor_class=CnnFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        agent = agent_fn(env=vec_env,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         policy_kwargs=policy_kwargs,
                         **self.learner_kwargs
                    )
        agent.set_logger(loggers)
        return agent

    def setup_loggers(self, log_dir):
        # if self.config.EXPERIMENT_PARAMETERS.independent:
        #     loggers = [configure(os.path.join(log_dir, f"agent_{i}"), ["stdout", "tensorboard", "csv"]) for i in range(self.config.ENV_PARAMETERS.num_agent)]
        # else:
        #     loggers = [configure(log_dir, ["stdout", "tensorboard", "csv"])]
        return [configure(log_dir, ["stdout", "tensorboard", "csv"])]

    def run(self):
        """Runs the experiment."""

        agent, callback, log_interval = self.init_experiment()

        print("Training starting...")
        agent.learn(total_timesteps=self.config.EXPERIMENT_PARAMETERS.total_timesteps,
                    log_interval=log_interval,
                    callback=callback,
                    progress_bar=True)
        print("Finished training.")

    @staticmethod
    def parse_args():
        """Parses command-line arguments."""
        import argparse
        parser = argparse.ArgumentParser(description="Experiment Runner")
        parser.add_argument("--config", "-c", help="Path to the configuration file", default='nrp.yaml')
        return parser.parse_args()

    @staticmethod
    def load_yaml(config_path):
        """
        Load a YAML configuration file, either from the provided path or from a default configs directory.

        Args:
            config_path (str): Path to the YAML file (absolute, relative, or filename).
            configs_dir (str): Directory to search for the file if not found in the provided path.

        Returns:
            dict: Loaded YAML configuration as a dictionary.

        Raises:
            FileNotFoundError: If the file cannot be found in both locations.
        """
        # Convert to Path object for cleaner path manipulation
        config_path = Path(config_path)

        # Case 1: Try to load the file as-is (absolute or relative path)
        if config_path.exists():
            print(f"Loading configuration from: {config_path}")
            with config_path.open("r") as file:
                return yaml.safe_load(file)

        # Case 2: Try to load the file from the configs directory
        fallback_path = Path(os.path.join(str(Path(__file__).resolve().parent.parent), "configs", config_path))
        if fallback_path.exists():
            print(f"Loading configuration from fallback directory: {fallback_path}")
            with fallback_path.open("r") as file:
                return yaml.safe_load(file)

        # Raise an error if the file cannot be found in either location
        raise FileNotFoundError(f"Configuration file '{config_path}' not found in either '{config_path}' or '{fallback_path}'.")

class NRPRunner(BaseRunner):
    def __init__(self, config_path):
        super().__init__(config_path)

        # Experiment settings
        self.experiment = self.config.EXPERIMENT_PARAMETERS.experiment
        self.experiment += '_independent' if self.config.EXPERIMENT_PARAMETERS.independent else ""

        self.experiment += '_penalty' if self.config.EXPERIMENT_PARAMETERS.penalty else ""
        self.log_dir = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results")

    def init_experiment(self):
        if self.config.RL_PARAMETERS.learner == "dqn":
            if self.config.EXPERIMENT_PARAMETERS.independent:
                self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
                agent_fn = IndependentDQN
            else: 
                agent_fn = DQN
        elif self.config.RL_PARAMETERS.learner == "ppo":
            agent_fn = PPO
        else:
            raise ValueError(f"Unsupported learner type: {self.learner}")
        
        return super().init_experiment(agent_fn)

class PRMRunner(BaseRunner):

    def __init__(self, config_path):
        super().__init__(config_path)

        # Experiment settings
        self.experiment = self.config.EXPERIMENT_PARAMETERS.experiment
        self.experiment += '_penalty' if self.config.EXPERIMENT_PARAMETERS.penalty else ""
        self.log_dir = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results")

    def init_experiment(self):
        """Configures the agent."""
        if self.config.EXPERIMENT_PARAMETERS.independent:
            self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
        agent_fn = RLWithRewardPredictor
        return super().init_experiment(agent_fn)    

    def set_up_rp(self, logger):
        """Set up reward predictor and its logging infrastructure."""
        n_labels = self.config.RRP_PARAMETERS.n_labels
        num_timesteps = self.config.EXPERIMENT_PARAMETERS.total_timesteps
        pretrain_labels = self.config.RP_PARAMETERS.pretrain_labels
        num_envs = self.config.ENV_PARAMETERS.num_envs
        train_freq = self.config.RP_PARAMETERS.train_freq
        buffer_ratio = self.config.RP_PARAMETERS.buffer_ratio
        pre_trian = self.config.RP_PARAMETERS.pre_trian
        predictor_epochs = self.config.RP_PARAMETERS.predictor_epochs
        lr = self.config.RP_PARAMETERS.lr

        # Configure logging
        predictor_loggers = AgentLoggerSb3(logger)

        # Set up the label schedule (if applicable)
        label_schedules = None
        if n_labels:
            label_schedules = LabelAnnealer(
                agent_logger=predictor_loggers,
                final_timesteps=num_timesteps,
                final_labels=n_labels,
                pretrain_labels=pretrain_labels
            )


        # Initialize reward predictor
        env = self.self.setup_environment(self.config, num_envs)
        predictor = ComparisonRewardPredictor(
            num_envs=num_envs,
            agent_loggers=predictor_loggers,
            label_schedules=label_schedules,
            observation_space=env.observation_space,
            action_space=env.action_space,
            epochs=predictor_epochs,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            lr=lr,
            train_freq=train_freq,
            comparison_collector_max_len=int(n_labels * buffer_ratio),
            pre_train=pre_trian,
            from_model=False,
            model_path="",
            freeze_rp=False
        )

        # Pre-train if necessary
        if pre_trian:
            env_factory = function_wrapper(self.setup_environment, self.config, 1)
            predictor.pre_trian(
                make_env=env_factory,
                pretrain_labels=pretrain_labels
            )

        return predictor

    def setup_loggers(log_dir):
        rl_logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
        rp_loggers = AgentLoggerSb3(rl_logger)
        loggers = {"rl": [rl_logger],
                   "rp": [rp_loggers]}
        return loggers

    def setup_agent(self, vec_env, agent_fn, log_dir):
        loggers = self.setup_loggers(log_dir)
        rp = self.set_up_rp(loggers["rp"])
        policy_kwargs = dict(
            features_extractor_class=CnnFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        agent = agent_fn(env=vec_env,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         policy_kwargs=policy_kwargs,
                         replay_buffer_class=RLWithRewardPredictorBuffer
                         **self.learner_kwargs
                    )
        agent.set_logger(loggers["rl"])
        return agent   

