import gymnasium
from torch import nn
import torch
import torch.nn.functional as F


class EmbedCnnNetwork(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        fcnet_hiddens=[1024, 128, 32],
        emb_dim=32,
        n_actions=2
    ):
        super(EmbedCnnNetwork, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_frames, view_len, _ =  observation_space.shape
        self.embed = nn.Embedding(n_actions, emb_dim, max_norm=0.1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_frames,  
                out_channels=16, 
                kernel_size=5,
                stride=1,
                padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(
                in_channels= 16,  # Input: (3 * 4) x 15 x 15
                out_channels=32,  # Output: 24 x 13 x 13
                kernel_size=3,
                stride=1,
                padding=0),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Flatten(start_dim=1)
        )
        with torch.no_grad():
            flat_out = self.conv(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.features = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.mlp = nn.Sequential(
        nn.Linear(fcnet_hiddens[0]+emb_dim, fcnet_hiddens[1]),
        nn.ReLU(),
        nn.Linear(fcnet_hiddens[1], fcnet_hiddens[2]),
        nn.ReLU(),
        nn.Linear(fcnet_hiddens[2], 1)
    )

        

    def forward(self, observations, actions) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        # observations = observations.permute(0, 3, 1, 2)
        # normalize the observation if it is not already
        if observations.max() > 1:
            observations = observations / 255.0

        # expand the observation if it is not already
        if len(observations.shape) == 3:
            # Need to add channels
            observations = torch.expand_dims(observations, axis=-1)

        emb = self.embed(actions.view(-1).long())
        x = self.conv(observations)
        x = F.relu(self.features(x))
        x = torch.cat([x, emb], axis=1)
        rewards = self.mlp(x)
        return rewards
    


class OneHotCnnNetwork(nn.Module):
    def __init__(self, observation_space=(3, 64, 64), n_actions=8, fcnet_hiddens=[], emb_dim=0):
        super(OneHotCnnNetwork, self).__init__()
        self.n_actions = n_actions
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # nn.LayerNorm(observation_space.shape),
            nn.Conv2d(observation_space.shape[0], 8, kernel_size=3, stride=1, padding=0),  
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((3, 3)),  # Global Average Pooling (keeps output fixed)
            nn.Flatten()
        )
        
        # Compute CNN output size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]
        print(f"cnn_output_dim: {cnn_output_dim}")
        # Linear Projection for One-Hot Encoded Actions
        self.action_projection = nn.Linear(n_actions, fcnet_hiddens[0]) 
        self.fetures = nn.Linear(cnn_output_dim , fcnet_hiddens[0])
        # Fully Connected Layers (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(fcnet_hiddens[0] * 2, fcnet_hiddens[1]),
            nn.ReLU(),
            nn.Linear(fcnet_hiddens[1], fcnet_hiddens[2]),
            nn.ReLU(),
            nn.Linear(fcnet_hiddens[2], 1)  # Output a single scalar
        )

    def forward(self, image, action):
        """
        image: Tensor of shape (batch_size, C, H, W)
        action: Tensor of shape (batch_size,) representing categorical action indices
        """

        image /= 255.0
        # Extract features from image
        image_features = F.relu(self.fetures(self.cnn(image)))
        
        # One-Hot Encode Action
        action_one_hot = F.one_hot(torch.tensor(action).squeeze().long(), num_classes=self.n_actions).float()  # Convert action to one-hot
        action_features = self.action_projection(action_one_hot)  # Project to feature space
        # Concatenate features
        combined_features = torch.cat((image_features, action_features), dim=1)
        
        # Predict scalar output
        return self.mlp(combined_features)