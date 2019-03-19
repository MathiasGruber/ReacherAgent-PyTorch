import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from libs.models import Actor, Critic
from libs.memory import ReplayBuffer, PrioritizedReplayMemory

BATCH_SIZE = 128        # Batch Size
BUFFER_SIZE = int(1e5)  # Memory capacity
GAMMA = 0.99            # Discount factor
LR_ACTOR = 1e-4         # Actor lerning rate
LR_CRITIC = 1e-3        # Critic learning rate
TAU = 1e-3              # Soft update of target networks
WEIGHT_DECAY = 0        # L2 weight decay for Critic


class Agents():
    
    def __init__(self, state_size, action_size, num_agents, random_state):
        """Initialize an Agent object.
        
        Arguments:
            state_size (int) -- dimension of each state
            action_size (int) -- dimension of each action
            num_agents (int) -- number of agents
            random_seed (int) -- random seed

        Keyword Arguments:        
            random_state {int} -- seed for torch random number generator (default: {42})
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random_state

        # Whether to use GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor Network (incl. target network)
        self.actor_local = Actor(state_size, action_size, random_state=self.seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_state=self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (incl. target network)
        self.critic_local = Critic(state_size, action_size, random_state=self.seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_state=self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), self.seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory. Use random sample from buffer to learn."""
        
        # Store experience
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # Train networks
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Determine agent action based on state
        
        Arguments:
            state {array_like} -- current state
        
        Keyword Arguments:
            add_noise {bool} -- whether to add Ornstain-Uhlenbeck noise to action-making
        
        Returns:
            [array-like] -- continous action(s) to take by agent
        """
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        """Reset the internal state of Ornstain-Uhlenbeck noise to mean"""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update actor and critic networks based on experience tuple.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Arguments:
            experiences (Tuple[torch.Tensor]) -- tuple of (s, a, r, s', done) 
            gamma (float) -- discount factor
        """

        # Unpack the tuple
        states, actions, rewards, next_states, dones = experiences

        # UPDATING THE CRITIC
        #####################

        # Get Q values from target models for next states
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # UPDATING THE ACTOR
        #####################

        # Compute loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # UPDATING TARGET NETWORKS
        ##########################
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Update the target model parameters (w_target) as follows:
        w_target = TAU*w_local + (1 - TAU)*w_target

        Arguments:
            local_model {PyTorch model} -- local model to copy from
            target_model {PyTorch model} -- torget model to copy to
            tau {float} -- interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, random_state, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(random_state)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()    

    def reset(self):
        """Reset the internal state to mean."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
