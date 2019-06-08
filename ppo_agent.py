import numpy as np
import random
from collections import namedtuple, deque

from models import QNetwork
from models import ActorPPO

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
BATCH_SIZE = 64         # batch size for learning from trajectory
TAU = 1e-3              # for soft update of target parameters
LR_CRIT = 1e-3          # learning rate critic
LR_ACTR = 1e-4          # learning rate actor
WEIGHT_DECAY = 1e-2     # L2 weight decay
GD_EPOCH = 5            # how often to optimize when learning is triggered


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
        self.qnetwork = QNetwork(state_size, action_size, hidden_layers, seed).to(device)
        self.qnetwork_optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR_CRIT, weight_decay=WEIGHT_DECAY)
        
        # mu-Network / Actor
        # Create the network, define the criterion and optimizer
        hidden_layers = [33, 33]
        self.munetwork = ActorPPO(state_size, action_size, hidden_layers, seed).to(device)
        self.munetwork_optimizer = optim.Adam(self.munetwork.parameters(), lr=LR_ACTR)
        
    
    def step(self, states, actions, rewards):        
        # Learn every UPDATE_EVERY time steps.
        # self.learn(experiences, GAMMA)
        
        # get discounts ready
        discount = GAMMA**np.arange(len(rewards))
        print(discount)
        print(discount[:,np.newaxis])
        print(np.asarray(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis] 
        print(np.asarray(rewards))
        
        # convert everything to torch 
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # check sizes
        print('Size states {}'.format(states.size()))
        print('Size actions {}'.format(actions.size()))
        print('Size rewards {}'.format(rewards.size()))

        # convert states to policy (or probability)
        _, dist = self.munetwork(states)         # get distribution
        log_probs = dist.log_prob(actions)      # get probability
        
        # get state values
        values = self.qnetwork(states, actions)
              
        # convert rewards to future rewards
        rewards_future = torch.empty(rewards.shape[0], rewards.shape[1], dtype=torch.float, device=device)
        for i in range(rewards.shape[0]):
            rewards_future[i,:] = torch.sum(rewards[i:,:],dim=0)
        print('Size future rewards {}'.format(rewards_future.size()))    
        print(rewards_future)
        
        # Compute TD error 
        values = self.qnetwork(states, actions)
        print('Size values {}'.format(values.size()))
        #td_error = reward + GAMMA * next_value - value
        # Compute advantages
        #advantage = advantage * TAU * GAMMA * done + td_error
        
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
        #if add_noise:
            #actions += beta*self.noise.sample()
        return np.clip(actions[0], -1, 1)
    
    def sample(self, states, actions, log_probs_old, rewards, advantages):
        idx = np.random.randint(0, states.size(0), BATCH_SIZE)
        return states[idx,:], actions[idx,:], log_probs_old[idx,:], rewards[idx,:], advantages[idx,:]
    
    
    def learn(self, trajectory, gamma):
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
        states, actions, rewards, next_states, dones = self.sample(trajectory)

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
            
    def clipped_surrogate(states, actions, log_probs_old, rewards_future, advantages, discount = 0.995, epsilon=0.1, beta=0.01):
        """Clipped surrogate function for PPO.
        
        Params
        ======
            policy  (PyTorch model): current policy that gets updated
            states  (PyTorch tensor): states of given trajectory for computation
            actions (PyTorch tensor): actions of given trajectory for computation
            rewards (PyTorch tensor): rewards of given trajectory for computation
            discount (float): discounting future rewards
            epsilon (float): clipping value
            beta (float): exploration factor for adding noise
        """
        # normalization
        mean = torch.mean(rewards_future, dim=1)
        std = torch.std(rewards_future, dim=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        # convert states to policy (or probability)
        _, dist = self.munetwork(states)
        new_probs = dist.log_prob(actions)
        
        ratio = (log_probs - log_probs_old).exp()
        #ratio = new_probs/old_probs

        # actual PPO comes here
        expected_return = torch.min(rewards_normalized*ratio, rewards_normalized*torch.clamp(ratio, 1-epsilon, 1+epsilon))

        return torch.mean(expected_return + beta*self.noise.sample())

