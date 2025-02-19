
from env import parallel_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecMonitor
import supersuit as ss
from buffers import CRMShardReplayBuffer
def setup_environment():
    """Configures the environment."""
    env = parallel_env(
        num_agents=2,
        ep_length=600,
        penalty=0,
        spawn_speed='fast',
        metric='Efficiency'
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, 2)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3"
    )
    env.get_attr = lambda x, y: ["human" for _ in range(env.num_envs)]
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env.get_full_state = lambda: env.env_method("get_full_state")
    return env

env = setup_environment()
env.reset()
a, b, c, d, e, = env.step([0, 1])
CRMShardReplayBuffer()