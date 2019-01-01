import numpy as np

from model import Actor, CriticD4PG
import torch
import torch.optim as optim
from OUNoise import OUNoise
from contextlib import contextmanager

@contextmanager
def evaluating(agent):
    """Temporarily switch to evaluation mode."""
    nets = [agent.actor_local, agent.actor_target, agent.critic_local, agent.critic_target]
    with torch.no_grad():
        istrain = [net.training for net in nets]
        try:
            for net in nets:
                net.eval()
            yield agent
        finally:
            for train_mode, net in zip(istrain, nets):
                if train_mode:
                    net.train()


class D4PG:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, device, num_atoms, q_min, q_max, lr_actor=1e-4, lr_critic=1e-4, weight_decay=0.0001):

        self.state_size = state_size
        self.action_size = action_size
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        # Critic Network (w/ Target Network)
        self.critic_local = CriticD4PG(state_size, action_size, seed, n_atoms=num_atoms, v_min=q_min, v_max=q_max).to(
            device)
        self.critic_target = CriticD4PG(state_size, action_size, seed, n_atoms=num_atoms, v_min=q_min, v_max=q_max).to(
            device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        self.noise = OUNoise(action_size)  # add OU noise for exploration
        # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        self.device = device

    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
            Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, state, add_noise=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # state = torch.tensor(np.moveaxis(state,3,1)).float().to(device)
        state = torch.tensor(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += add_noise * self.noise.noise()
        return np.clip(action, -1.0, 1.0)
