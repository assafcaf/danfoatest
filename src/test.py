from reward_predictor import parallel_collect_segments,function_wrapper
from env import parallel_env
import supersuit as ss
from stable_baselines3.common.vec_env import VecTransposeImage, VecMonitor


def parrallel_env(num_envs=1, num_agents=1, num_frames=2, ep_length=600, penalty=False, spawn_speed='slow'):
    """Configures the environment."""
    env = parallel_env(
        num_agents=num_agents,
        ep_length=ep_length,
        penalty=penalty,
        spawn_speed=spawn_speed
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=(num_envs // 4) if (num_envs // 4) > 0 else 1,
        base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    return env


env_factory = function_wrapper(parrallel_env, num_envs=1, num_agents=1, num_frames=2, ep_length=600)
paths = parallel_collect_segments(env_factory=env_factory,
                                    n_desired_segments=8,
                                    segment_length=600,
                                    workers=4,
                                    num_agents=1)
x=1


