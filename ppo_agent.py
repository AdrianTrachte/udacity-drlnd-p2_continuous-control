import numpy as np
import random
from collections import namedtuple, deque

from models import CriticPPO
from models import ActorPPO

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

GAMMA = 1.0             # discount factor
LAMBDA = 0.5            # Value for Generalized Advantage Estimation, lambda = 0 --> 1 step TD estimate
BATCH_SIZE = 64         # batch size for learning from trajectory
LR_CRIT = 1e-3          # learning rate critic
LR_ACTR = 1e-4          # learning rate actor
WEIGHT_DECAY = 1e-2     # L2 weight decay
GD_EPOCH = 20           # how often to optimize when learning is triggered
EPS_CLIP = 0.2          # PPO clipping value
GRAD_CLIP = 2           # clipping of gradient for optimization

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

        # Critic
        # Create the network, define the criterion and optimizer
        hidden_layers = [37, 37]
        self.vnetwork = CriticPPO(state_size, 1, hidden_layers, seed).to(device)
        self.vnetwork_optimizer = optim.Adam(self.vnetwork.parameters(), lr=LR_CRIT, weight_decay=WEIGHT_DECAY)
        
        # mu-Network / Actor
        # Create the network, define the criterion and optimizer
        hidden_layers = [33, 33]
        self.munetwork = ActorPPO(state_size, action_size, hidden_layers, seed).to(device)
        self.munetwork_optimizer = optim.Adam(self.munetwork.parameters(), lr=LR_ACTR)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)
    
    def step(self, states, actions, rewards, beta, infos=False):        
        """Step agent and compute advantage function for learning.
        
        Params
        ======
            states (array_like): states of trajectory
            actions (array_like): actions of trajectory
            rewards (array_like): rewards of trajectory
        """
        # convert everything to torch 
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # compute advantages
        #advantages = self.compute_gae(states, rewards)
        advantages = self.compute_tderror(states, rewards)        # use td-error as advantage function
        
        # normalization
        advantages = (advantages - advantages.mean()) / advantages.std()

        # compute g return for critic training now with discounted rewards
        discount = (GAMMA)**torch.arange(len(rewards), dtype=torch.float, device=device)
        rewards = rewards*torch.unsqueeze(discount,1)    # rewards discounted

        # convert rewards to future rewards
        rewards_future = torch.empty(rewards.shape[0], rewards.shape[1], dtype=torch.float, device=device)
        for i in range(rewards.shape[0]):
            rewards_future[i,:] = torch.sum(rewards[i:,:],dim=0)
        
        if infos:
            self.show_infos(states, actions, rewards, rewards_future, advantages)

        # trigger learning   
        self.learn(states, actions, rewards_future, advantages, beta)
        
    def show_infos(self, states, actions, rewards, rewards_future, advantages):
        print("\n")
        """
        print("states size: {}".format(states.size()))
        print("actions size: {}".format(actions.size()))
        print("rewards size: {}".format(rewards.size()))
        print("rewards_future size: {}".format(rewards_future.size()))
        print("advantages size: {}".format(advantages.size()))
        """
        print("------------------------------")
        print("rewards sum: {}".format(rewards.sum()))
        
        fig = plt.figure(figsize=(12,8))
        fig.subplots_adjust(hspace=.4)

        plt.subplot(2,2,1)
        plt.plot(rewards.cpu().numpy())
        plt.title('rewards')
        
        plt.subplot(2,2,2)
        plt.plot(rewards_future.cpu().numpy())
        plt.title('rewards_future')
        
        plt.subplot(2,2,3)
        values = self.vnetwork(states)
        plt.plot(values.detach().cpu().numpy())
        plt.title('values')
        
        plt.subplot(2,2,4)
        plt.plot(advantages.detach().cpu().numpy())
        plt.title('advantages')
        plt.show()
            
    def compute_gae(self, states, rewards):
        """ compute the generalized advantage estimation
        
        Params
        ======
            states (PyTorch tensor): states of trajectory
            rewards (PyTorch tensor): rewards of trajectory
        """
        # compute td_error
        td_error = self.compute_tderror(states, rewards)
        
        # get discounts and lambda ready
        discount = (GAMMA*LAMBDA)**torch.arange(len(rewards), dtype=torch.float, device=device)
        # compute advantage function
        advantages = torch.empty(rewards.shape[0], rewards.shape[1], dtype=torch.float, device=device)
        for i in range(advantages.shape[0]):
            if i == 0:
                advantages[i,:] = torch.sum(td_error[i:,:]*torch.unsqueeze(discount[:],1),dim=0)
            else:
                advantages[i,:] = torch.sum(td_error[i:,:]*torch.unsqueeze(discount[0:-i],1),dim=0)
        return advantages        
        
    def compute_tderror(self, states, rewards):
        """ compute the td-error
        
        Params
        ======
            states (PyTorch tensor): states of trajectory            
        """
        # Compute advantage function TD error 
        values = self.vnetwork(states)   # get state values
        next_values = torch.zeros(values.shape[0],values.shape[1], dtype=torch.float, device=device)
        next_values[:-1] = values[1:]    # next values, last next_value is zero
        # compute actual td-error
        # td_error = r_t + GAMMA * V(s_t+1) - V(s_t)
        td_error = rewards + GAMMA*next_values - values
        return td_error
        
    def act(self, state, beta = 1.0, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            add_noise (boolean): add noise for exploration
        """
        state = torch.from_numpy(state).float().to(device)
        self.munetwork.eval()
        with torch.no_grad():
            mu, dist = self.munetwork(state)
        self.munetwork.train()
        actions = dist.sample().cpu().data.numpy()
        if add_noise:
            actions += beta*self.noise.sample()
        return np.clip(actions[0], -1, 1)
    
    def learn(self, states, actions, rewards_future, advantages, beta):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            states (PyTorch tensor): states of trajectory
            actions (PyTorch tensor): actions of trajectory
            rewards_future (PyTorch tensor): all future rewards of trajectory
            advantages (PyTorch tensor): advantage function of trajectory
        """
        
        # convert states to policy (or probability)
        _, dist = self.munetwork(states)        # get distribution
        log_probs_old = dist.log_prob(actions)      # get probability

        
        for _ in range(GD_EPOCH):
            for _ in range(states.size(0) // BATCH_SIZE):
                states_s, actions_s, log_probs_old_s, rewards_future_s, advantages_s = self.sample(states, actions, log_probs_old, rewards_future, advantages)

                # ---------------------------- update critic ---------------------------- #
                # value loss
                values = self.vnetwork(states_s)   # get state values
                value_loss = F.mse_loss(rewards_future_s, values)
                # Minimize the loss
                self.vnetwork_optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.vnetwork.parameters(), GRAD_CLIP)
                value_loss.backward()
                self.vnetwork_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actor_loss = self.clipped_surrogate(states_s, actions_s, log_probs_old_s, advantages_s, beta=beta)
                # Minimize the loss
                self.munetwork_optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.munetwork.parameters(), GRAD_CLIP)
                actor_loss.backward(retain_graph=True)
                self.munetwork_optimizer.step()
            
    def sample(self, states, actions, log_probs_old, rewards, advantages):
        idx = np.random.randint(0, states.size(0), BATCH_SIZE)
        return states[idx,:], actions[idx,:], log_probs_old[idx,:], rewards[idx,:], advantages[idx,:]
            
    def clipped_surrogate(self, states, actions, log_probs_old, advantages, beta=0.01):
        """Clipped surrogate function for PPO.
        
        Params
        ======
            states  (PyTorch tensor): states of given trajectory for computation
            actions (PyTorch tensor): actions of given trajectory for computation
            advantages (PyTorch tensor): advantage function of given trajectory for computation
            beta (float): exploration factor for adding noise
        """
        # convert states to policy (or probability)
        _, dist = self.munetwork(states)
        log_probs = dist.log_prob(actions)
        
        ratio = (log_probs - log_probs_old).exp()
        #ratio = new_probs/old_probs

        # actual PPO comes here
        expected_return = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP)) - beta*dist.entropy()

        return torch.mean(expected_return)
    
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
