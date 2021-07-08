import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.distributions import Normal, Categorical

# dict to enable loading activations based on a string
nn_activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


def mlp_hidden_layers(input_dim, hidden_sizes, activation="relu"):
    """ Helper function to create hidden MLP layers.
    N.B: The same activation is applied after every layer

    Args:
        input_dim: An int denoting the input size of the mlp
        hidden_sizes: A list with ints containing hidden sizes
        activation: A str specifying the activation function

    Returns:

    """
    activation = nn_activations[activation]
    dims = [input_dim] + hidden_sizes
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
    return layers


def orthogonal_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class MLPGaussianActor(nn.Module):
    def __init__(self, cfg, device):
        super(MLPGaussianActor, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=cfg["obs_dim"], hidden_sizes=cfg["mlp"]["hidden_sizes"],
                                   activation=cfg["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.mu = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], cfg["action_dim"])
        log_std = -0.5 * np.ones(cfg["action_dim"], dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
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


class Critic(nn.Module):
    def __init__(self, cfg, device):
        super(Critic, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=cfg["obs_dim"], hidden_sizes=cfg["mlp"]["hidden_sizes"],
                                   activation=cfg["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.value = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], 1)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
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

        layers = mlp_hidden_layers(input_dim=cfg["obs_dim"], hidden_sizes=cfg["mlp"]["hidden_sizes"],
                                   activation=cfg["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)

        self.mu = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], cfg["action_dim"])
        self.log_std = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], cfg["action_dim"])

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
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
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi


class MLPQFunction(nn.Module):
    def __init__(self, cfg, device):
        super(MLPQFunction, self).__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim=cfg["obs_dim"] + cfg["action_dim"],
                                   hidden_sizes=cfg["mlp"]["hidden_sizes"], activation=cfg["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.q = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], 1)

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        obs = obs.to(self.device)
        action = action.to(self.device)
        phi = self.phi(torch.cat([obs, action], dim=-1))
        q = self.q(phi)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

    def get_features(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            phi = self.phi(x)
        return phi

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

        layers = mlp_hidden_layers(input_dim=cfg["obs_dim"] + cfg["action_dim"],
                                   hidden_sizes=cfg["mlp"]["hidden_sizes"], activation=cfg["mlp"]["activation"])
        self.phi = nn.Sequential(*layers)
        self.action_logits = nn.Linear(cfg["mlp"]["hidden_sizes"][-1], cfg["action_dim"])

        # Orthogonal Weight Initialization
        self.apply(orthogonal_weight_init)
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
