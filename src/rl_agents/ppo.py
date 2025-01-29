from typing import Any, Dict, List, Optional, Type
import numpy as np
from stable_baselines3 import PPO as sb3_PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

import psutil 

class PPO(sb3_PPO):

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_ep_metrics = info.get("metrics")
            maybe_ep_fire = {"fire": info.get("fire")}
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info | maybe_ep_metrics| maybe_ep_fire])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
    
    def _dump_logs(self, iteration):
        """
        Collect statistics from learning and export it to an internal logger
        :param episode_logg: Dictionary of <Tag (str): statistic values (List)>
        """
        self.logger.record("metrics/efficiency", safe_mean([ep_info["efficiency"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/equality", safe_mean([ep_info["equality"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/sustainability", safe_mean([ep_info["sustainability"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/peace", safe_mean([ep_info["peace"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/fire_attempts", safe_mean([ep_info["fire_attempts"] for ep_info in self.ep_info_buffer]))
        self.logger.record("metrics/fire_sucsses", safe_mean([ep_info["fire_sucsses"] for ep_info in self.ep_info_buffer]))
        super()._dump_logs(iteration)
