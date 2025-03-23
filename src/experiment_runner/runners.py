
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# local imports
from callbacks import SingleAgentCallback
from configs import Config
from reward_predictor import AgentLoggerSb3, RPMRewardPredictor, CRMRewardPredictor
from learners import CollectiveRLRPLearner, IndependentRLRPLearner
from buffers import PRMShardReplayBuffer, CRMShardReplayBuffer, PRMShardRolloutBuffer
from env import parallel_env
from rl_agents import DQN, IndependentDQN, PPO, IndependentPPO, CnnFeatureExtractor, CustomCNN, DQNPRM, DQNCRM, PPOPRM

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
        if self.config.RL_PARAMETERS.learner == "dqn":
            self.learner_kwargs = {
                "learning_rate": self.config.RL_PARAMETERS.learning_rate,
                "batch_size": self.config.RL_PARAMETERS.batch_size,
                "tau" :self.config.RL_PARAMETERS.tau,
                "gamma": self.config.RL_PARAMETERS.gamma,
                "train_freq": self.config.RL_PARAMETERS.train_freq,
                "exploration_fraction": self.config.RL_PARAMETERS.exploration_fraction,
                "learning_starts": self.config.RL_PARAMETERS.learning_starts,
                "buffer_size": self.compute_buffer_size(),
                "exploration_final_eps": self.config.RL_PARAMETERS.exploration_final_eps,
                "policy": self.config.RL_PARAMETERS.policy,
                'replay_buffer_class': PRMShardReplayBuffer,
                'replay_buffer_kwargs': {"episode_length": self.config.ENV_PARAMETERS.ep_length}
            }
        else: 
            self.learner_kwargs = {
            "learning_rate": self.config.RL_PARAMETERS.learning_rate,
            "batch_size": self.config.RL_PARAMETERS.batch_size,
            "ent_coef" :self.config.RL_PARAMETERS.ent_coef,
            "gamma": self.config.RL_PARAMETERS.gamma,
            "gae_lambda": self.config.RL_PARAMETERS.gae_lambda,
            "target_kl": self.config.RL_PARAMETERS.target_kl,
            "max_grad_norm": self.config.RL_PARAMETERS.max_grad_norm,
            "policy": self.config.RL_PARAMETERS.policy,
            "n_steps": self.config.RL_PARAMETERS.n_steps,
            "n_epochs": self.config.RL_PARAMETERS.n_epochs,
            "vf_coef": self.config.RL_PARAMETERS.vf_coef,
            "clip_range": self.config.RL_PARAMETERS.clip_range,
            "rollout_buffer_class": PRMShardRolloutBuffer,
            "rollout_buffer_kwargs": {"replay_kwargs": {"episode_length": self.config.ENV_PARAMETERS.ep_length,
                                                        "buffer_size": self.compute_buffer_size()}}
            }
                # Experiment settings
        
        self.experiment = self.config.EXPERIMENT_PARAMETERS.experiment+f"-{self.config.RL_PARAMETERS.learner}"
        self.experiment += '-independent' if self.config.EXPERIMENT_PARAMETERS.independent else ""
        self.experiment += '-penalty' if self.config.EXPERIMENT_PARAMETERS.penalty else ""

    def init_experiment(self, agent_fn):
        dir_name = self.config.EXPERIMENT_PARAMETERS.run_name_template.format(
            experiment=self.experiment,
            metric=self.config.EXPERIMENT_PARAMETERS.metric,
            spawn_speed=self.config.RL_PARAMETERS.spawn_speed,
            num_agents=self.config.ENV_PARAMETERS.num_agent
        )
        self.config.EXPERIMENT_PARAMETERS.run_name_template = dir_name
        log_dir = os.path.join(self.log_dir, dir_name, "run_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        vec_env = self.setup_environment(self.config, self.config.ENV_PARAMETERS.num_envs)
        eval_env = self.setup_environment(self.config, 1)
        agent = self.setup_agent(vec_env, agent_fn, log_dir)

        render_frequency = self.config.RL_PARAMETERS.render_frequency*self.config.ENV_PARAMETERS.ep_length
        if self.config.RL_PARAMETERS.learner == "ppo":
            log_interval = 1
            render_frequency = self.config.RL_PARAMETERS.render_frequency
        else:
            log_interval = self.config.ENV_PARAMETERS.num_agent * self.config.ENV_PARAMETERS.num_envs
            render_frequency = self.config.RL_PARAMETERS.render_frequency*self.config.ENV_PARAMETERS.ep_length

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
            metric=config.EXPERIMENT_PARAMETERS.metric,
            ascii_map=config.ENV_PARAMETERS.map,
            agent_view_range=config.ENV_PARAMETERS.agent_view_range
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
        env.get_attr = lambda x, y: ["human" for _ in range(env.num_envs)]
        env = VecMonitor(env)
        env = VecTransposeImage(env)
        
        # Expose the get_full_state function from the underlying environment
        if config.EXPERIMENT_PARAMETERS.experiment == "crm":
            # env.get_full_states = lambda: [e.par_env.env.aec_env.env.env.env.get_full_state() for e in env.venv.venv.venv.vec_envs]
            # env.state_space = env.venv.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.state_space
            env.get_full_state = env.env_method("get_full_state")
            env.state_space = env.env_method("state_space")
        return env

    def setup_agent(self, vec_env, agent_fn, log_dir):
        loggers = self.setup_loggers(log_dir)
        
        # # # ssd paper
        # policy_kwargs = dict(
        #     features_extractor_class=CustomCNN,
        #     features_extractor_kwargs=dict(features_dim=self.config.RL_PARAMETERS.features_dim,
        #                                    num_frames=self.config.ENV_PARAMETERS.num_frames),
        #     net_arch=[self.config.RL_PARAMETERS.features_dim],
        # )
        
        # my network
        policy_kwargs = dict(
            features_extractor_class=CnnFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=self.config.RL_PARAMETERS.features_dim),
        )
        agent = agent_fn(env=vec_env,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        #  policy_kwargs=policy_kwargs,
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
        total_timesteps = (self.config.EXPERIMENT_PARAMETERS.n_episodes
                           *self.config.ENV_PARAMETERS.ep_length
                           *self.config.ENV_PARAMETERS.num_envs
                           *self.config.ENV_PARAMETERS.num_agent)
        print("Training starting...")
        if type(agent) is CollectiveRLRPLearner or  type(agent) is IndependentRLRPLearner:
            agent.learn(
                learner=self.config.RL_PARAMETERS.learner,
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                callback=callback,
                progress_bar=True
            )
        else:
            agent.learn(
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                callback=callback,
                progress_bar=True
            )
        print("Finished training.")
        exit()

    def compute_buffer_size(self):
        
        buffer_size = self.config.RL_PARAMETERS.buffer_size
        num_envs = self.config.ENV_PARAMETERS.num_envs
        num_agent = self.config.ENV_PARAMETERS.num_agent
        ep_length = self.config.ENV_PARAMETERS.ep_length
        
        # if not independent, the buffer size multiplies by the number of agents
        if not self.config.EXPERIMENT_PARAMETERS.independent: 
            buffer_size = buffer_size*num_envs*num_agent*ep_length
        else: # if independent, the buffer size not multiplied by the number of agents
            buffer_size = buffer_size*num_envs*ep_length
        print(f"buffer_size: {buffer_size}")
        return buffer_size

    @staticmethod
    def parse_args():
        """Parses command-line arguments."""
        import argparse
        parser = argparse.ArgumentParser(description="Experiment Runner")
        parser.add_argument("--config", "-c", help="Path to the configuration file", default='prm.yaml')
        return parser.parse_args()

    @staticmethod
    def load_yaml(config_path):
        """
        Load a YAML configuration file, either from the provided path or from a default configs directory.

        Args:
            config_path (str): Path to the YAML file (absolute, relative, or filename).
        Returns:
            dict: Loaded YAML configuration as a dictionary.

        Raises:
            FileNotFoundError: If the file cannot be found in both locations.
        """
        try:
            if config_path.split(".")[-1] != "yaml":
                config_path += ".yaml"
        except: 
            pass

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
        self.log_dir = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results")

    def init_experiment(self):
        if self.config.RL_PARAMETERS.learner == "dqn":
            if self.config.EXPERIMENT_PARAMETERS.independent:
                self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
                agent_fn = IndependentDQN
            else: 
                agent_fn = DQN
        elif self.config.RL_PARAMETERS.learner == "ppo":
            if self.config.EXPERIMENT_PARAMETERS.independent:
                self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
                agent_fn = IndependentPPO
            else:
                agent_fn = PPO
        else:
            raise ValueError(f"Unsupported learner type: {self.learner}")
        
        # remove keys that belong to prm anget
        del  self.learner_kwargs['replay_buffer_class']
        del  self.learner_kwargs['replay_buffer_kwargs']
        return super().init_experiment(agent_fn)

class PRMRunner(BaseRunner):

    def __init__(self, config_path):
        super().__init__(config_path)

        # Experiment settings
        self.log_dir = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results")

    def init_experiment(self):
        if self.config.RL_PARAMETERS.learner == "dqn":
            if self.config.EXPERIMENT_PARAMETERS.independent:
                self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
                agent_fn = IndependentDQN
            else: 
                agent_fn = DQNPRM
        elif self.config.RL_PARAMETERS.learner == "ppo":
            agent_fn = PPOPRM
        else:
            raise ValueError(f"Unsupported learner type: {self.learner}")
        
        return super().init_experiment(agent_fn) 

    def set_up_rp(self, logger, env):
        """Set up reward predictor and its logging infrastructure."""
        predictor_epochs = self.config.RP_PARAMETERS.predictor_epochs
        lr = self.config.RP_PARAMETERS.lr


        # Initialize reward predictor
        predictor = RPMRewardPredictor(
            agent_logger=logger,
            observation_space=env.observation_space,
            n_actions=env.action_space.n,
            epochs=predictor_epochs,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            lr=lr,
            network=self.config.RP_PARAMETERS.network,
            network_kwargs={
                'emb_dim':self.config.RP_PARAMETERS.emb_dim, 
                'fcnet_hiddens': self.config.RP_PARAMETERS.fcnet_hiddens
            }
        )
        return predictor

    def setup_loggers(self, log_dir):
        rl_logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
        rp_loggers = AgentLoggerSb3(rl_logger)
        loggers = {"rl": [rl_logger],
                   "rp": [rp_loggers]}
        return loggers

    def setup_agent(self, vec_env, agent_fn, log_dir):
        loggers = self.setup_loggers(log_dir)
        reward_predictor = self.set_up_rp(loggers["rp"], vec_env)
        train_rp_freq = self.config.RP_PARAMETERS.train_freq*self.config.ENV_PARAMETERS.ep_length
        policy_kwargs = dict(
            features_extractor_class=CnnFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        rl_agent = agent_fn(env=vec_env,
                            predictor=reward_predictor,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            policy_kwargs=policy_kwargs,
                            **self.learner_kwargs
                    )
        rl_agent.set_logger(loggers["rl"])
        agent = CollectiveRLRPLearner(rl_agent=rl_agent,
                                      rp_learning_starts=self.config.RP_PARAMETERS.learning_starts,
                                      reward_predictor=reward_predictor,
                                      train_rp_freq=train_rp_freq,
                                      async_rp_training=False,
                                      parallel_agents=False,
                                      batch_size=self.config.RP_PARAMETERS.batch_size*2)

        return agent   

class CRMRunner(BaseRunner):
    def __init__(self, config_path):
        super().__init__(config_path)

        # Experiment settings
        self.log_dir = os.path.join(str(Path(__file__).resolve().parent.parent.parent), "results")

    def init_experiment(self):
        self.learner_kwargs["num_agents"] = self.config.ENV_PARAMETERS.num_agent
        agent_fn = DQNCRM
        return super().init_experiment(agent_fn) 

    def set_up_rp(self, logger, env):
        """Set up reward predictor and its logging infrastructure."""
        predictor_epochs = self.config.RP_PARAMETERS.predictor_epochs
        lr = self.config.RP_PARAMETERS.lr


        # Initialize reward predictor
        predictor = CRMRewardPredictor(
            agent_logger=logger,
            num_outputs = self.config.ENV_PARAMETERS.num_agent,
            observation_space=env.state_space,
            action_space=env.action_space.n,
            epochs=predictor_epochs,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            lr=lr,
        )
        return predictor

    def setup_loggers(self, log_dir):
        rl_logger = configure(log_dir, ["stdout", "tensorboard", "csv"])
        rp_loggers = AgentLoggerSb3(rl_logger)
        loggers = {"rl": [rl_logger],
                   "rp": [rp_loggers]}
        return loggers

    def setup_agent(self, vec_env, agent_fn, log_dir):
        loggers = self.setup_loggers(log_dir)
        reward_predictor = self.set_up_rp(loggers["rp"], vec_env)
        train_rp_freq = self.config.RP_PARAMETERS.train_freq*self.config.ENV_PARAMETERS.ep_length
        policy_kwargs = dict(
            features_extractor_class=CnnFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        rl_agent = agent_fn(env=vec_env,
                            predictor=reward_predictor,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            policy_kwargs=policy_kwargs,
                            replay_buffer_class=CRMShardReplayBuffer,
                            replay_buffer_kwargs={"episode_length": self.config.ENV_PARAMETERS.ep_length,
                                                  "state_space": vec_env.state_space,
                                                  "n_agents": self.learner_kwargs["num_agents"]
                                                  },
                            **self.learner_kwargs
                    )
        rl_agent.set_logger(loggers["rl"])
        agent = IndependentRLRPLearner(rl_agent=rl_agent,
                                      rp_learning_starts=self.config.RP_PARAMETERS.learning_starts,
                                      reward_predictor=reward_predictor,
                                      train_rp_freq=train_rp_freq,
                                      async_rp_training=False,
                                      parallel_agents=False,
                                      batch_size=self.config.RP_PARAMETERS.batch_size*2)

        return agent   