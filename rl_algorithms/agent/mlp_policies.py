import torch
import numpy as np
import torch.nn.functional as F

from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions import Normal, Categorical


class MLPGaussianActor(nn.Module):
    def __init__(self, cfg, device):
        super(MLPGaussianActor, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu = nn.Linear(64, cfg["action_dim"])
        log_std = -0.5 * np.ones(cfg["action_dim"], dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        mu = self.mu(phi)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def lprob(self, x, a):
        x = x.to(self.device)
        dist = self._dist(x)
        # Last axis sum needed for Torch Normal distribution
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device)).sum(axis=-1)
        return lprob, dist

    def compute_action(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            dist = self._dist(x)
            action = dist.sample()
            lprob = dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), lprob.cpu().item()


class Actor(nn.Module):
    """ Discrete MLP Actor """
    def __init__(self, cfg, device):
        super(Actor, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.action_logits = nn.Linear(32, cfg["action_dim"])
        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        action_prob_logits = self.action_logits(phi)
        return Categorical(logits=action_prob_logits)

    def lprob(self, x, a):
        x = x.to(self.device)
        dist = self._dist(x)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device))
        return lprob, dist

    def compute_action(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            dist = self._dist(x)
            action = dist.sample()
            lprob = dist.log_prob(action)
        return action.cpu().item(), lprob.cpu().item()


class Critic(nn.Module):
    def __init__(self, cfg, device):
        super(Critic, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value = nn.Linear(64, 1)
        self.to(device=device)

    def forward(self, x):
        x = x.to(self.device)
        phi = self.phi(x)
        return self.value(phi).view(-1)


class SquashedGaussianMLPActor(nn.Module):
    """ Continous MLP Actor for Soft Actor-Critic """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, cfg, device):
        super(SquashedGaussianMLPActor, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, cfg["action_dim"])
        self.log_std = nn.Linear(256, cfg["action_dim"])

        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def forward(self, x, with_lprob=True):
        x = x.to(self.device)
        dist = self._dist(x)
        action = dist.rsample()
        if with_lprob:
            lprob = dist.log_prob(action).sum(axis=-1)
            lprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            lprob = None
        action = torch.tanh(action)
        return action, lprob

    def get_features(self, x):
        x = x.to(self.device)
        return self.phi(x)


class MLPQFunction(nn.Module):
    def __init__(self, cfg, device):
        super(MLPQFunction, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"] + cfg["action_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.q = nn.Linear(256, 1)
        self.to(device=device)

    def forward(self, obs, action):
        obs = obs.to(self.device)
        action = action.to(self.device)
        phi = self.phi(torch.cat([obs, action], dim=-1))
        q = self.q(phi)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

    def get_features(self, x):
        x = x.to(self.device)
        return self.phi(x)


class SACPolicy(nn.Module):
    def __init__(self, cfg, device):
        super(SACPolicy, self).__init__()
        self.device = device

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(cfg=cfg, device=device)
        self.q1 = MLPQFunction(cfg=cfg, device=device)
        self.q2 = MLPQFunction(cfg=cfg, device=device)
        self.to(device)

    def compute_action(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.pi(obs, with_lprob=False)
            return action.cpu().numpy()

    def critic(self, obs, action):
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2


class MLPDiscreteActor(nn.Module):
    def __init__(self, cfg, device):
        super(MLPDiscreteActor, self).__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(cfg["obs_dim"], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.action_logits = nn.Linear(32, cfg["action_dim"])
        self.to(device=device)

    def _dist(self, x):
        phi = self.phi(x)
        action_prob_logits = self.action_logits(phi)
        return Categorical(logits=action_prob_logits)

    def lprob(self, x, a):
        x = x.to(self.device)
        dist = self._dist(x)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device))
        return lprob, dist

    def compute_action(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            dist = self._dist(x)
            action = dist.sample()
            lprob = dist.log_prob(action)
        return action.cpu().item(), lprob.cpu().item()
