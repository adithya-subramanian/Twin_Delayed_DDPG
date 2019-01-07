import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-5       # L2 weight decay
UPDATE_DELAY = 2         # Delay for policy update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed = 0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, self.seed).to(device)                  # Initializing the local Actor model
        self.actor_target = Actor(state_size, action_size, self.seed).to(device)                 # Initializing the target Actor model
        self.critic_1_local = Critic(state_size, action_size, self.seed).to(device)              # Initializing the local 1st Critic model
        self.critic_1_target = Critic(state_size, action_size, self.seed).to(device)             # Initializing the target 1st Critic model
        self.critic_2_local = Critic(state_size, action_size, self.seed).to(device)              # Initializing the local 2nd Critic model
        self.critic_2_target = Critic(state_size, action_size, self.seed).to(device)             # Initializing the target 2nd Critic model
        self.actor_optimizer = optim.RMSprop(self.actor_local.parameters(), lr=LR_ACTOR,weight_decay = WEIGHT_DECAY)         # Initializing the optimizer for Actor model's parameters
        self.critic_1_optimizer = optim.RMSprop(self.critic_1_local.parameters(), lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)  # Initializing the optimizer for 1st Critic model's parameters
        self.critic_2_optimizer = optim.RMSprop(self.critic_2_local.parameters(), lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)  # Initializing the optimizer for 2nd Critic model's parameters
        # Noise process
        self.noise = OUNoise(action_size, random_seed) # Initializing the noise distribution

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) # Initializing the replay buffer
    
    def step(self, state, action, reward, next_state, done,step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)    # Adding an experience to the replay buffer

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:                          
            experiences = self.memory.sample()                      # Sampling episode from the replay buffer
            self.learn(experiences, GAMMA,step)                     # Calls the learn function responsible for loss computation and updating the model's parameter

    def act(self,state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()    # Compute the action vector for the given state from Actor
            #print(action)
        self.actor_local.train()
        if add_noise:
            #print(self.noise.sample())
            action += self.noise.sample()                          # Add noise to the action vector
        return np.clip(action, -1, 1)                              # Clamp dimension's of the noise within the range

    def reset(self):
        self.noise.reset()                                         # Reset the noise parameters to the values to mean and sigma

    def learn(self, experiences, gamma,step):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            step (int): number of steps
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)                                                           # Compute the action vector for the given  next state from the target Actor
        actions_next = actions_next + torch.from_numpy(self.noise.sample()).type(torch.FloatTensor).to(device)  # Adding Noise to the target action vector
        actions_next = torch.clamp(actions_next,min = -1,max = 1)                                               # Clipping the dimension of the action within the range
        Q_1_targets_next = self.critic_1_target(next_states, actions_next)                                      # Compute Q - values using Q_1 for the next state 
        Q_2_targets_next = self.critic_2_target(next_states, actions_next)                                      # Compute Q - values using Q_2 for the next state
        # Compute Q targets for current states (y_i)
        Q_targets_1 = rewards + (gamma * torch.min(Q_1_targets_next,Q_2_targets_next) * (1 - dones))            # Compute Q - target using minimum between Q_1 and Q_2 for the current state         
        # Compute critic loss
        Q_1_expected = self.critic_1_local(states, actions)                                                     # Compute Q - expected for the current state using Q_1
        Q_2_expected = self.critic_2_local(states, actions)                                                     # Compute Q - expected for the current state using Q_2
        critic_1_loss = F.mse_loss(Q_1_expected, Q_targets_1)                                                   # Compute the critic loss for expected Q1 value 
        critic_2_loss = F.mse_loss(Q_2_expected, Q_targets_1)                                                   # Compute the critic loss for expected Q1 value
        # Minimize the loss
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward(retain_graph = True)
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if step%UPDATE_DELAY == 0:                                                                              # Delayed Actor parameters update 
	        # Compute actor loss
	        actions_pred = self.actor_local(states)
	        actor_loss = -self.critic_1_local(states, actions_pred).mean()
	        # Minimize the loss
	        self.actor_optimizer.zero_grad()
	        actor_loss.backward()
	        self.actor_optimizer.step()
	        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_1_local, self.critic_1_target, TAU)                                        # Soft update of the target networks 
        self.soft_update(self.critic_2_local, self.critic_2_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)