import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    """Dueling Double Deep Q Network Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size,400)
        self.fc2 = nn.Linear(400,300)
        self.fc3 = nn.Linear(300,4)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()

    def forward(self,x):
        x = self.act2(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act1(self.fc3(x))
        return x

class Critic(nn.Module):
    """Dueling Double Deep Q Network Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size+action_size,400)
        self.fc2 = nn.Linear(action_size + 400,300)
        self.fc3 = nn.Linear(300,1)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()

    def forward(self,x,y):
        x = self.act2(self.fc1(torch.cat([x,y],dim = 1)))
        x = self.act2(self.fc2(torch.cat([x,y],dim = 1)))
        x = self.act1(self.fc3(x))
        return x