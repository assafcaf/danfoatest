import torch
from torch import nn
import numpy as np


############################################## Reward Predictors networks ##############################################    
class MlpHead(nn.Module):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, h_size=64, emb_dim=32, n_actions=2, num_outputs=1):
        super(MlpHead, self).__init__()
        input_dim = np.prod(obs_shape)
        self.double()
        self.embed = nn.Embedding(n_actions, emb_dim, max_norm=1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+emb_dim, h_size),
            nn.LeakyReLU(),
            nn.Linear(h_size, num_outputs)
        )

    def forward(self, obs, act):
        flat_obs = obs.flatten(1)
        emb = self.embed(act.long())
        x = torch.cat([flat_obs, emb], axis=1)
        return self.mlp(x)
    
    def copy(self):
        """Return a copy of the model."""
        copy = MlpHead(self.observation_space, self.h_size, self.emb_dim, self.n_actions, self.num_outputs)
        copy.load_state_dict(self.state_dict())
        return copy 


class MlpNetwork(MlpHead):

    def __init__(self, observation_space, features_dim=128, h_size=16, emb_dim=8, n_actions=2, num_outputs=1):
        super().__init__(features_dim, h_size, emb_dim, n_actions, num_outputs)
        self.observation_space = observation_space
        self.features_dim = features_dim
        self.h_size = h_size
        self.emb_dim = emb_dim
        self.n_actions = n_actions
        self.num_outputs = num_outputs
       
        with torch.no_grad():
            n_flatten = torch.flatten(torch.as_tensor(observation_space.sample())).shape[0]
            
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(n_flatten, features_dim*4),
                                    nn.LeakyReLU(0.01),
                                    nn.Linear(features_dim*4, features_dim),
                                    nn.LeakyReLU(0.01))
        self.float()

    def copy(self):
        """Return a copy of the model."""
        copy = MlpNetwork(self.observation_space, self.features_dim, self.h_size, self.emb_dim, self.n_actions, self.num_outputs)
        copy.load_state_dict(self.state_dict())
        return copy

    def forward(self, obs, act):
        # normalize the observation if it is not already
        if obs.max() > 1:
            obs = obs / 255.0
        
        # expand the observation if it is not already
        if len(obs.shape) == 3:
            # Need to add channels
            obs = torch.expand_dims(obs, axis=-1)
            
        emb = self.embed(act.view(-1).long())
        x = self.linear(obs)
        x = torch.cat([x, emb], axis=1)
        return self.mlp(x)
    

class CnnNetwork(MlpHead):
    """
    Network that has two convolution steps on the observation space before flattening,
    concatinating the action and being an MLP.
    """

    def __init__(self, observation_space, features_dim=32, h_size=16, emb_dim=8, n_actions=2, num_outputs=1):
        super().__init__(features_dim, h_size, emb_dim, n_actions, num_outputs)
        self.observation_space = observation_space
        self.features_dim = features_dim
        self.h_size = h_size
        self.emb_dim = emb_dim
        self.n_actions = n_actions
        self.num_outputs = num_outputs
        # my backbonde
        self.back_bone = nn.Sequential(
                            nn.Conv2d(observation_space.shape[0], 16, kernel_size=5, stride=1, padding=0),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.Dropout2d(0.2),
                            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.Dropout2d(0.2),
                            nn.Flatten(),
                    )
        
        with torch.no_grad():
            n_flatten = self.back_bone(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.LayerNorm(n_flatten),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self.float()
    
    def copy(self):
        """Return a copy of the model."""
        copy = CnnNetwork(self.observation_space, self.features_dim, self.h_size, self.emb_dim, self.n_actions, self.num_outputs)
        copy.load_state_dict(self.state_dict())
        return copy
    
    def forward(self, obs, act):
        # normalize the observation if it is not already
        if obs.max() > 1:
            obs = obs / 255.0
        
        # expand the observation if it is not already
        if len(obs.shape) == 3:
            # Need to add channels
            obs = torch.expand_dims(obs, axis=-1)
            
        emb = self.embed(act.view(-1).long())
        x = self.linear(self.back_bone(obs))
        x = torch.cat([x, emb], axis=1)
        return self.mlp(x)
