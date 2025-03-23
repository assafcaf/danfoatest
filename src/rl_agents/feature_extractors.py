from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import Space
import gymnasium
from torch import nn
import torch
import torch.nn.functional as F

class CnnFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 128, fcnet_hiddens = [256, 128]):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
                        nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=1, padding="valid"),
                        nn.ReLU(),
                        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="valid"),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(in_features=n_flatten, out_features=fcnet_hiddens[0]),
            nn.ReLU(),
            nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1]),
            nn.ReLU(),)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim=128,
        view_len=10,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        # observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features