import torch
import time
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import SubsetRandomSampler, DataLoader


class Reinforce(object):
    def __init__(self, cfg, pi, buffer, device=torch.device('cpu')):
        self.pi = pi
        self.buffer = buffer
        self.cfg = cfg
        self.gamma = cfg['gamma']
        self.batch_size = cfg['batch_size']
        self.device = device
        self.opt = Adam(self.pi.parameters(), lr=cfg['lr'], weight_decay=cfg['l2_reg'])
        self.n_updates = 0

    def push_and_update(self, obs, action, reward, log_prob, done, next_obs=None):
        self.buffer.push(obs, action, reward, log_prob, done)
        if len(self.buffer) >= self.batch_size and done:
            tic = time.time()
            self.update(next_obs)
            self.buffer.reset()
            self.n_updates += 1
            print("Update {} took {}s".format(self.n_updates, time.time()-tic))

    def estimate_returns(self, rewards, dones):
        rets = torch.as_tensor(np.zeros_like(rewards, dtype=np.float32), device=self.device)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + (1 - dones[t]) * (self.gamma * running_add)
            rets[t] = running_add
        return rets

    def update(self, next_obs):
        observations, actions, rewards, _, dones = self.buffer.sample(len(self.buffer))
        observations, actions, rewards, dones = observations.to(self.device), actions.to(self.device), \
                                                            rewards.to(self.device), dones.to(self.device)
        rets = self.estimate_returns(rewards, dones)
        lprobs, _ = self.pi.lprob(observations, actions)

        loss = (-lprobs * rets).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class BatchActorCritic(Reinforce):
    def __init__(self, cfg, pi, critic, buffer, device=torch.device('cpu')):
        super(BatchActorCritic, self).__init__(cfg=cfg, pi=pi, buffer=buffer, device=device)
        self.lmbda = self.cfg["lmbda"]
        self.critic = critic
        self.critic_loss = nn.MSELoss()
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg['lr'], weight_decay=cfg['l2_reg'])

    def estimate_returns_advantages(self, rewards, dones, vals):
        """ len(rewards) = len(dones) = len(vals)-1

        Args:
            rewards:
            dones:
            vals:

        Returns:

        """
        advs = torch.as_tensor(np.zeros(len(vals), dtype=np.float32), device=self.device)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (1 - dones[t]) * self.gamma * vals[t+1] - vals[t]
            advs[t] = delta + (1 - dones[t]) * self.gamma * self.lmbda * advs[t+1]

        rets = advs[:-1] + vals[:-1]
        return rets, advs[:-1]

    def update(self, next_obs):
        observations, actions, rewards, _, dones = self.buffer.sample(len(self.buffer))
        observations, actions, rewards, dones = observations.to(self.device), actions.to(self.device), \
                                                            rewards.to(self.device), dones.to(self.device)
        lprobs, _ = self.pi.lprob(x=observations, a=actions)
        vals = self.critic(torch.cat([
            observations, torch.as_tensor(next_obs.astype(np.float32), device=self.device).view(1, -1)]))
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        actor_loss = -(lprobs * advs).mean()
        critic_loss = self.critic_loss(vals[:-1], rets)
        loss = actor_loss + critic_loss
        self.opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.critic_opt.step()


class PPO(BatchActorCritic):
    """ PPO-Clip """
    def __init__(self, cfg, pi, critic, buffer, device=torch.device('cpu')):
        super(PPO, self).__init__(cfg=cfg, pi=pi, critic=critic, buffer=buffer, device=device)
        self.n_epochs = cfg["n_epochs"]
        self.opt_batch_size = cfg["opt_batch_size"]
        self.clip_epsilon = cfg["clip_epsilon"]

    def push(self, obs, action, reward, log_prob, done):
        self.buffer.push(obs, action, reward, log_prob, done)
        return len(self.buffer) >= self.batch_size and done

    def ppo_update(self, next_obs):
        tic = time.time()
        self.update(next_obs)
        self.buffer.reset()
        self.n_updates += 1
        print("Update {} took {}s".format(self.n_updates, time.time() - tic))

    def update(self, next_obs):
        observations, actions, rewards, old_lprobs, dones = self.buffer.sample(len(self.buffer))
        observations, actions, rewards, old_lprobs, dones = observations.to(self.device), actions.to(self.device), \
                                                            rewards.to(self.device), old_lprobs.to(self.device), \
                                                            dones.to(self.device)
        with torch.no_grad():
            vals = self.critic(torch.cat([
                observations, torch.as_tensor(next_obs.astype(np.float32), device=self.device).view(1, -1)]))
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        # Normalize advantages
        norm_advs = (advs - advs.mean()) / advs.std()

        inds = np.arange(len(rewards))
        for itr in range(self.n_epochs):
            np.random.shuffle(inds)
            for i_start in range(0, len(self.buffer), self.opt_batch_size):
                opt_inds = inds[i_start: min(i_start+self.opt_batch_size, len(inds)-1)]
                # Policy update preparation
                new_lprobs, _ = self.pi.lprob(observations[opt_inds], actions[opt_inds])
                new_vals = self.critic(observations[opt_inds])
                ratio = torch.exp(new_lprobs - old_lprobs[opt_inds])
                p_loss = ratio * norm_advs[opt_inds]
                clipped_p_loss = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * norm_advs[opt_inds]
                actor_loss = -(torch.min(p_loss, clipped_p_loss)).mean()
                critic_loss = self.critic_loss(new_vals, rets[opt_inds])
                loss = actor_loss + critic_loss

                # Apply gradients
                self.opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.critic_opt.step()


class PPOPenalty(PPO):
    """ PPO with Adaptive KL-Penalty """
    def __init__(self, cfg, pi, critic, buffer, device=torch.device('cpu')):
        super(PPOPenalty, self).__init__(cfg=cfg, pi=pi, critic=critic, buffer=buffer, device=device)
        self.target_kl = cfg["target_kl"]
        self.beta = 1

    def update(self, next_obs):
        observations, actions, rewards, old_lprobs, dones = self.buffer.sample(len(self.buffer))
        observations, actions, rewards, old_lprobs, dones = observations.to(self.device), actions.to(self.device), \
                                                            rewards.to(self.device), old_lprobs.to(self.device), \
                                                            dones.to(self.device)
        with torch.no_grad():
            _, old_dists = self.pi.lprob(observations, actions)
            old_probs = old_dists.probs
            vals = self.critic(torch.cat([
                observations, torch.as_tensor(next_obs.astype(np.float32), device=self.device).view(1, -1)]))
        rets, advs = self.estimate_returns_advantages(rewards=rewards, dones=dones, vals=vals)

        # Normalize advantages
        norm_advs = (advs - advs.mean()) / advs.std()

        inds = np.arange(len(rewards))
        for itr in range(self.n_epochs):
            np.random.shuffle(inds)
            for i_start in range(0, len(self.buffer), self.opt_batch_size):
                opt_inds = inds[i_start: min(i_start+self.opt_batch_size, len(inds)-1)]
                # Policy update preparation
                new_lprobs, new_dists = self.pi.lprob(observations[opt_inds], actions[opt_inds])
                new_vals = self.critic(observations[opt_inds])
                ratio = torch.exp(new_lprobs - old_lprobs[opt_inds])
                p_loss = -(ratio * norm_advs[opt_inds]).mean()

                kl = F.kl_div(old_probs[opt_inds], new_dists.probs)

                # if kl > 1.5 * self.target_kl:
                #     self.beta = 2 * self.beta
                # if kl < self.target_kl / 1.5:
                #     self.beta = self.beta / 2

                actor_loss = p_loss + self.beta * kl
                critic_loss = self.critic_loss(new_vals, rets[opt_inds])
                loss = actor_loss + critic_loss

                # Apply gradients
                self.opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.critic_opt.step()
