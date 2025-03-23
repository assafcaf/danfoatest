
from typing import NamedTuple
import numpy as np

class PRMEpisodeData(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    experiment_rewards: np.ndarray
    true_rewards:np.ndarray
    aip:np.array

    def __add__(self, other):
        """Dynamically concatenates PRMEpisodeData instances without relying on hardcoded field names."""
        combined_data = {}

        for field in self._fields:  # Iterate over all named fields
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            # If both values are not None, concatenate them
            if self_value is not None and other_value is not None:
                combined_data[field] = np.concatenate([self_value, other_value], axis=0)
            elif self_value is not None:
                combined_data[field] = self_value
            else:
                combined_data[field] = other_value

        return PRMEpisodeData(**combined_data)

    def __len__(self):
        return self.observations.shape[0]


class CRMEpisodeData(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    experiment_rewards: np.ndarray
    true_rewards:np.ndarray
    aip:np.array

    def __add__(self, other):
        """Dynamically concatenates PRMEpisodeData instances without relying on hardcoded field names."""
        combined_data = {}

        for field in self._fields:  # Iterate over all named fields
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            # If both values are not None, concatenate them
            if self_value is not None and other_value is not None:
                combined_data[field] = np.concatenate([self_value, other_value], axis=0)
            elif self_value is not None:
                combined_data[field] = self_value
            else:
                combined_data[field] = other_value

        return PRMEpisodeData(**combined_data)

    def __len__(self):
        return self.states.shape[0]

