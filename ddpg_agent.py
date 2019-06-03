import numpy as np
import random
from collections import namedtuple, deque

from models import QNetwork
from models import ActorPolicy

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size for learning
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_CRIT = 1e-3          # learning rate critic
LR_ACTR = 1e-4          # learning rate actor
WEIGHT_DECAY = 1e-2     # L2 weight decay
UPDATE_EVERY = 1        # how often to update the network
GD_EPOCH = 1            # how often to optimize when learning is triggered


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network / Critic
        # Create the network, define the criterion and optimizer
        hidden_layers = [37, 37]
        # hidden_layers = [37, 37, 37]
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, seed).to(device)
        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR_CRIT, weight_decay=WEIGHT_DECAY)
        
        # mu-Network / Actor
        # Create the network, define the criterion and optimizer
        hidden_layers = [33, 33]
        # hidden_layers = [33, 33, 33]
        self.munetwork_local = ActorPolicy(state_size, action_size, hidden_layers, seed).to(device)
        self.munetwork_target = ActorPolicy(state_size, action_size, hidden_layers, seed).to(device)
        self.munetwork_optimizer = optim.Adam(self.munetwork_local.parameters(), lr=LR_ACTR)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, beta = 1.0, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            add_noise (boolean): add noise for exploration
        """
        state = torch.from_numpy(state).float().to(device)
        self.munetwork_local.eval()
        with torch.no_grad():
            actions = self.munetwork_local(state).cpu().data.numpy()
        self.munetwork_local.train()
        if add_noise:
            actions += beta*self.noise.sample()
        return np.clip(actions, -1, 1)
    
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) = mu(state) -> action
            critic_target(state, action) = Q(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        for _ in range(GD_EPOCH):
            # Get predicted next-state actions and Q values from target models
            actions_next = self.munetwork_target(next_states)
            Q_targets_next = self.qnetwork_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.qnetwork_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.qnetwork_optimizer.zero_grad()
            #torch.nn.utils.clip_grad_norm(self.qnetwork_local.parameters(), 1)
            critic_loss.backward()
            self.qnetwork_optimizer.step()
            del critic_loss

        # ---------------------------- update actor ---------------------------- #
        for _ in range(GD_EPOCH):
            actions_pred = self.munetwork_local(states)
            # Compute actor loss
            actor_loss = -self.qnetwork_local(states, actions_pred).mean()
            # Minimize the loss
            self.munetwork_optimizer.zero_grad()
            actor_loss.backward()
            self.munetwork_optimizer.step()
            del actor_loss

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.munetwork_local, self.munetwork_target, TAU)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)      
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process.
    
    Params
    ======
        mu (float):  is the center point of the distribution
        theta (float): daws the distribution towards the center point
        sigma (float): draws the distribution towards random points of normal distribution
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
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