import gymnasium
from collections import defaultdict

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

def average_nested_dicts(dicts, key):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for d in dicts:
        for k, v in d.get(key, {}).items():
            totals[k] += v
            counts[k] += 1

    return {k: totals[k] / counts[k] for k in totals}