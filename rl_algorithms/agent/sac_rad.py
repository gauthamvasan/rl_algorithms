"""
Adaptation of Yufeng Yuan's async SAC RAD implementation by Yan Wang & Gautham Vasan:
- https://github.com/YufengYuan/ur5_async_rl/blob/main/sac_rad.py
- https://github.com/Yan-Wang88/visual-reacher-sac/blob/master/algo/sac_rad_agent.py
"""

import queue
import torch
import copy

import rl_algorithms.agent.utils as utils
import numpy as np
import torch.multiprocessing as mp

from rl_algorithms.agent.replay_buffer import RadReplayBuffer, AsyncRadReplayBuffer
from rl_algorithms.agent.cnn_policies import ActorModel, CriticModel


class SacRadAgent:
    """SAC algorithm."""

    def __init__(
            self,
            image_shape,
            proprioception_shape,
            action_shape,
            device,
            net_params,
            discount=0.99,
            init_temperature=0.1,
            alpha_lr=1e-3,
            actor_lr=1e-3,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_tau=0.005,
            rad_offset=0.01,
            async_mode=False,
            replay_buffer_capacity=100000,
            batch_size=256,
            update_every=50,
            update_epochs=50,
            max_updates_per_step=10,
            init_steps=1000,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.async_mode = async_mode
        self.update_every = update_every
        self.update_epochs = update_epochs
        self.init_steps = init_steps

        if not 'conv' in net_params:  # no image
            image_shape = (0, 0, 0)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        self.action_dim = action_shape[0]

        self.actor = ActorModel(image_shape, proprioception_shape, action_shape[0], net_params, rad_offset).to(device)

        self.critic = CriticModel(image_shape, proprioception_shape, action_shape[0], net_params, rad_offset).to(device)

        self.critic_target = copy.deepcopy(self.critic)  # also copies the encoder instance

        if hasattr(self.actor.encoder, 'convs'):
            self.actor.encoder.convs = self.critic.encoder.convs
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.num_updates = 0

        # optimizers
        self.init_optimizers()

        self.train()
        self.critic_target.train()

        if async_mode:
            self.share_memory()

            # initialize processes in 'spawn' mode, required by CUDA runtime
            ctx = mp.get_context('spawn')

            MAX_QSIZE = 2
            self.sample_queue = ctx.Queue(MAX_QSIZE)
            self.update_stat_queue = ctx.Queue(MAX_QSIZE)
            self.minibatch_queue = ctx.Queue(MAX_QSIZE)

            # initialize data augmentation process
            self.replay_buffer_process = ctx.Process(target=AsyncRadReplayBuffer,
                                                     args=(
                                                         image_shape,
                                                         proprioception_shape,
                                                         action_shape,
                                                         replay_buffer_capacity,
                                                         batch_size,
                                                         self.sample_queue,
                                                         self.minibatch_queue,
                                                         init_steps,
                                                         max_updates_per_step
                                                     )
                                                     )
            self.replay_buffer_process.start()
            # initialize SAC update process
            self.update_process = ctx.Process(target=self.async_update)
            self.update_process.start()

        else:
            self.replay_buffer = RadReplayBuffer(
                image_shape=image_shape,
                proprioception_shape=proprioception_shape,
                action_shape=action_shape,
                capacity=replay_buffer_capacity,
                batch_size=batch_size)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.log_alpha.share_memory_()

    def init_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=(0.9, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(0.5, 0.999)
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_action(self, image, state, deterministic=False):
        with torch.no_grad():
            if image is not None:
                image = torch.FloatTensor(image).to(self.device)
                image.unsqueeze_(0)

            if state is not None:
                state = torch.FloatTensor(state).to(self.device)
                state.unsqueeze_(0)

            mu, pi, _, _ = self.actor(
                image, state, random_rad=False, compute_pi=True, compute_log_pi=False,
            )

            if deterministic:
                return mu.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy().flatten()

    def update_critic(self, images, states, actions, rewards, next_images, next_states, dones):
        with torch.no_grad():
            _, policy_actions, log_pis, _ = self.actor(next_images, next_states)
            target_Q1, target_Q2 = self.critic_target(next_images, next_states, policy_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pis
            target_Q = rewards + ((1.0 - dones) * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(images, states, actions, detach_encoder=False)

        critic_loss = torch.mean((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        critic_stats = {
            'train_critic/loss': critic_loss.item()
        }

        return critic_stats

    def update_actor_and_alpha(self, images, states):
        # detach encoder, so we don't update it with the actor loss
        _, pis, log_pis, log_stds = self.actor(images, states, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(images, states, pis, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pis - actor_Q).mean()

        entropy = 0.5 * log_stds.shape[1] * (1.0 + np.log(2 * np.pi)
                                             ) + log_stds.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pis - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        actor_stats = {
            'train_actor/loss': actor_loss.item(),
            'train_actor/target_entropy': self.target_entropy.item(),
            'train_actor/entropy': entropy.mean().item(),
            'train_alpha/loss': alpha_loss.item(),
            'train_alpha/value': self.alpha.item(),
            'train/entropy': entropy.mean().item(),
        }
        return actor_stats

    def push_sample(self, image, state, action, reward, next_image, next_state, done):
        if self.async_mode:
            self.sample_queue.put((image, state, action, reward, next_image, next_state, done))
        else:
            self.replay_buffer.add(image, state, action, reward, next_image, next_state, done)

    def update_networks(self, step):
        if self.async_mode:
            try:
                stat = self.update_stat_queue.get_nowait()
            except queue.Empty:
                return {}

            return stat

        if step >= self.init_steps and (step % self.update_every == 0):
            for _ in range(self.update_epochs):
                stat = self.update(*self.replay_buffer.sample())

            return stat

        return {}

    def update(self, images, states, actions, rewards, next_images, next_states, dones):
        # regular update of SAC_RAD, sequentially augment data and train
        if images is not None:
            images = torch.as_tensor(images, device=self.device).float()
            next_images = torch.as_tensor(next_images, device=self.device).float()
        if states is not None:
            states = torch.as_tensor(states, device=self.device).float()
            next_states = torch.as_tensor(next_states, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        stats = self.update_critic(images, states, actions, rewards, next_images, next_states, dones)
        if self.num_updates % self.actor_update_freq == 0:
            actor_stats = self.update_actor_and_alpha(images, states)
            stats = {**stats, **actor_stats}
        if self.num_updates % self.critic_target_update_freq == 0:
            self.soft_update_target()
        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self.num_updates
        self.num_updates += 1

        return stats

    def async_update(self):
        while True:
            try:
                self.update_stat_queue.put_nowait(self.update(*self.minibatch_queue.get()))
            except queue.Full:
                pass

    def soft_update_target(self):
        utils.soft_update_params(
            self.critic.Q1, self.critic_target.Q1, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.Q2, self.critic_target.Q2, self.critic_tau
        )
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder,
            self.encoder_tau
        )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

    def close(self):
        if self.async_mode:
            self.replay_buffer_process.terminate()
            self.update_process.terminate()
            self.replay_buffer_process.join()
            self.update_process.join()

        del self
