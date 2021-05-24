"""
An adaptation of the OpenAI Spinning Up Soft Actor Critic implementation
Original implementation: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
"""

import torch
import time
import itertools

from torch import nn
from copy import deepcopy
from torch.optim import Adam, SGD


class SAC():
    def __init__(self, cfg, actor_critic, device=torch.device('cpu')):
        self.cfg = cfg
        self.actor_critic = actor_critic
        self.device = device

        self.gamma = cfg['gamma']
        self.alpha = cfg['alpha']
        self.batch_size = cfg['batch_size']
        self.polyak = cfg['polyak']
        self.steps_between_updates = cfg['steps_between_updates']
        self.n_epochs = cfg['n_epochs']
        self.n_warmup = cfg['n_warmup']

        # Create actor-critic module and target networks
        self.ac_target = deepcopy(actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        if hasattr(self.actor_critic, 'q1'):
            self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())
        else:
            # Support for shared network
            self.q_params = itertools.chain(self.actor_critic.critic.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=cfg["lr"])
        self.q_optimizer = Adam(self.q_params, lr=cfg["lr"])

        self.q1_loss = nn.MSELoss()
        self.q2_loss = nn.MSELoss()

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, observations, actions, rewards, dones, next_observations):
        q1, q2 = self.actor_critic.critic(observations, actions)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(next_observations)
            # Target Q-values
            q1_pi_targ, q2_pi_targ = self.ac_target.critic(next_observations, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rewards + self.gamma * (1 - dones) * (q_pi_targ - self.alpha * logp_a2)
        # MSE loss against Bellman backup
        loss_q1 = self.q1_loss(q1, backup)
        loss_q2 = self.q2_loss(q2, backup)
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())
        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, observations):
        pi, logp_pi = self.actor_critic.pi(observations)
        q1_pi, q2_pi = self.actor_critic.critic(observations, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, observations, actions, rewards, dones, next_observations):
        observations, actions, rewards, dones, next_observations = \
            observations.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), \
            next_observations.to(self.device)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(observations=observations, actions=actions, rewards=rewards,
                                             dones=dones, next_observations=next_observations)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(observations=observations)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.ac_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
