import time

import torch
import numpy as np

from collections import namedtuple
from torch.utils.data import Dataset
from threading import Lock


class SimpleBuffer(Dataset):
    Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'log_prob', 'done'))
    def __init__(self):
        self.buffer = []
        self.done_indices = []

    def push(self, obs, action, reward, log_prob, done):
        """ Saves a transition."""
        self.buffer.append(self.Transition(obs, action, reward, log_prob, done))
        if done:
            self.done_indices.append(len(self.buffer))

    def sample(self, batch_size):
        if batch_size >= len(self.buffer):
            batch = self.Transition(*zip(*self.buffer))
        else:
            raise NotImplemented

        observations = torch.from_numpy(np.stack(batch.obs).astype(np.float32))
        actions = torch.from_numpy(np.stack(batch.action).astype(np.float32))
        rewards = torch.from_numpy(np.stack(batch.reward).astype(np.float32))
        dones = torch.from_numpy(np.stack(batch.done).astype(np.float32))
        lprobs = torch.from_numpy(np.stack(batch.log_prob).astype(np.float32))
        return observations, actions, rewards, lprobs, dones

    @property
    def n_episodes(self):
        return len(self.done_indices)

    def reset(self):
        self.buffer = []
        self.done_indices = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        t = self.Transition(*zip(*self.buffer[item]))
        return t.obs, t.action, t.reward, t.log_prob, t.done


class VisuomotorBuffer(Dataset):
    Transition = namedtuple('Transition', ('img', 'sensory', 'action', 'reward', 'log_prob', 'done'))

    def __init__(self, sensory_dim):
        """ Store images as uint8 to save RAM space. Strict assumptions about data structures are made here:
        - obs is a 1-D vector
        - obs[: sensory_dim] contains the sensory observation (i.e., excluding image)
        - obs[sensory_dim:] contains the flattened image

        Args:
            sensory_dim:
        """
        self.sensory_dim = sensory_dim
        self.buffer = []
        self.done_indices = []

    def push(self, obs, action, reward, log_prob, done):
        """ Saves a transition."""
        sensory_obs = obs[:self.sensory_dim]
        img = obs[self.sensory_dim:] * 255
        img = img.astype(np.uint8)

        self.buffer.append(self.Transition(img, sensory_obs, action, reward, log_prob, done))
        if done:
            self.done_indices.append(len(self.buffer))

    def sample(self, batch_size):
        if batch_size >= len(self.buffer):
            batch = self.Transition(*zip(*self.buffer))
        else:
            raise NotImplemented

        img = torch.from_numpy(np.stack(batch.img).astype(np.float32)) / 255.
        sensory_obs = torch.from_numpy(np.stack(batch.sensory).astype(np.float32))
        observations = torch.cat((img, sensory_obs), dim=-1) # Concatenate along columns
        actions = torch.from_numpy(np.stack(batch.action).astype(np.float32))
        rewards = torch.from_numpy(np.stack(batch.reward).astype(np.float32))
        dones = torch.from_numpy(np.stack(batch.done).astype(np.float32))
        lprobs = torch.from_numpy(np.stack(batch.log_prob).astype(np.float32))
        return observations, actions, rewards, lprobs, dones

    @property
    def n_episodes(self):
        return len(self.done_indices)

    def reset(self):
        self.buffer = []
        self.done_indices = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        t = self.Transition(*zip(*self.buffer[item]))
        return np.concatenate((t.img, t.sensory), -1), t.action, t.reward, t.log_prob, t.done


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.lock = Lock()

    def store(self, obs, action, reward, done, next_obs):
        with self.lock:
            self.observations[self.ptr] = obs
            self.next_observations[self.ptr] = next_obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        with self.lock:
            idxs = np.random.randint(0, self.size, size=batch_size)
            observations = torch.from_numpy(self.observations[idxs])
            next_observations = torch.from_numpy(self.next_observations[idxs])
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])
            return (observations, actions, rewards, dones, next_observations)

    def __len__(self):
        return self.size


class SACVisuomotorReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, img_dim, sensory_dim, act_dim, size):
        self.sensory_dim = sensory_dim
        self.images = np.zeros((size, np.prod(img_dim)), dtype=np.uint8)
        self.sensory_obs = np.zeros((size, sensory_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.lock = Lock()

    def store(self, obs, action, reward, done, next_obs):
        with self.lock:
            img = obs[self.sensory_dim:] * 255
            self.images[self.ptr] = img
            self.sensory_obs[self.ptr] = obs[:self.sensory_dim]
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        with self.lock:
            idxs = np.random.randint(0, self.size - 1, size=batch_size)
            img = torch.from_numpy(self.images[idxs].astype(np.float32)) / 255
            sensory_obs = torch.from_numpy(self.sensory_obs[idxs].astype(np.float32))
            observations = torch.cat((img, sensory_obs), dim=-1)  # Concatenate along columns
            actions = torch.from_numpy(self.actions[idxs])
            rewards = torch.from_numpy(self.rewards[idxs])
            dones = torch.from_numpy(self.dones[idxs])

            idxs += 1
            next_img = torch.from_numpy(self.images[idxs].astype(np.float32)) / 255
            next_sensory_obs = torch.from_numpy(self.sensory_obs[idxs].astype(np.float32))
            next_observations = torch.cat((next_img, next_sensory_obs), dim=-1)  # Concatenate along columns

        return (observations, actions, rewards, dones, next_observations)

    def __len__(self):
        return self.size
