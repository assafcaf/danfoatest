import math
from multiprocessing import Pool, Manager
import numpy as np
from tqdm import tqdm

def random_action(env):
    """ Pick a random action from the environment's action space. """
    return env.action_space.sample()

def do_rollout(env, action_function, segment_length, num_agents):
    """ Perform a rollout using the provided action function. """
    obs, rewards, actions = [], [], []
    ob = env.reset()
    
    for _ in range(segment_length):  # Use environment's max steps
        action = [action_function(env) for _ in range(num_agents)]
        obs.append(ob)
        actions.append(action)
        ob, reward, done, _ = env.step(action)
        rewards.append(reward)
        
        if done:
            break

    return {
        "obs": np.array(obs),
        "rewards": np.array(rewards),
        "actions": np.array(actions),
    }

def sample_segment(path, segment_length):
    """ Sample a random segment of the specified length from the path. """
    if len(path["obs"]) < segment_length:
        return None

    start_idx = np.random.randint(0, len(path["obs"]) - segment_length + 1)
    end_idx = start_idx + segment_length

    return {
        "obs": path["obs"][start_idx:end_idx],
        "rewards": path["rewards"][start_idx:end_idx],
        "actions": path["actions"][start_idx:end_idx],
    }

def collect_segments(env, n_desired_segments, segment_length, num_agents, progress_queue=None):
    """ Collect a specified number of segments from random rollouts. """
    segments = []
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action, segment_length, num_agents)
        num_segments = max(1, len(path["obs"]) // segment_length // 4)
        
        for _ in range(num_segments):
            segment = sample_segment(path, segment_length)
            if segment:
                segments.append(segment)
                if progress_queue:
                    progress_queue.put(1)  # Notify progress
                if len(segments) >= n_desired_segments:
                    break

    return segments[:n_desired_segments]

def parallel_collect_segments(env_factory, n_desired_segments, segment_length, workers, num_agents):
    """ Collect segments in parallel using multiple workers. """
    if workers < 2:
        return collect_segments(env_factory(), n_desired_segments, segment_length, num_agents)

    segments_per_worker = int(math.ceil(n_desired_segments / workers))
    with Manager() as manager:
        progress_queue = manager.Queue()
        total_progress = tqdm(total=n_desired_segments, desc="Collecting segments")

        def update_progress():
            while True:
                try:
                    progress_queue.get(timeout=0.1)
                    total_progress.update(1)
                except:
                    break

        with Pool(workers) as pool:
            jobs = [
                (env_factory, segments_per_worker, segment_length, num_agents, progress_queue)
                for _ in range(workers)
            ]
            results = pool.starmap_async(worker_collect_segments, jobs)

            while not results.ready():
                update_progress()

            segments = [segment for result in results.get() for segment in result]

        total_progress.close()
    return segments

def worker_collect_segments(env_factory, n_segments, segment_length, num_agents, progress_queue):
    """ Worker function to collect segments. """
    env = env_factory()
    return collect_segments(env, n_segments, segment_length, num_agents, progress_queue)
