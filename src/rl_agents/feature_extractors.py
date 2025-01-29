from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import Space
from torch import nn
import torch

class CnnFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 0):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
                        nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))