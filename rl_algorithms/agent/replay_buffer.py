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


class RadReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.ignore_image = True
        self.ignore_state = True

        if image_shape[-1] != 0:
            self.images = np.empty((capacity, *image_shape), dtype=np.uint8)
            self.next_images = np.empty((capacity, *image_shape), dtype=np.uint8)
            self.ignore_image = False

        if proprioception_shape[-1] != 0:
            self.states = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.next_states = np.empty((capacity, *proprioception_shape), dtype=np.float32)
            self.ignore_state = False

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.count = 0

    def add(self, image, state, action, reward, next_image, next_state, done):
        if not self.ignore_image:
            self.images[self.idx] = image
            self.next_images[self.idx] = next_image
        if not self.ignore_state:
            self.states[self.idx] = state
            self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.count = self.capacity if self.full else self.idx

    def sample(self):
        idxs = np.random.randint(
            0, self.count, size=min(self.count, self.batch_size)
        )
        if self.ignore_image:
            images = None
            next_images = None
        else:
            images = self.images[idxs]
            next_images = self.next_images[idxs]

        if self.ignore_state:
            states = None
            next_states = None
        else:
            states = self.states[idxs]
            next_states = self.next_states[idxs]

        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return images, states, actions, rewards, next_images, next_states, dones


class AsyncRadReplayBuffer(RadReplayBuffer):
    def __init__(self, image_shape, proprioception_shape, action_shape, capacity, batch_size,
                 sample_queue, minibatch_queue, init_steps, max_updates_per_step):
        super(AsyncRadReplayBuffer, self).__init__(image_shape, proprioception_shape, action_shape, capacity, batch_size)
        self.init_steps = init_steps
        self.step = 0
        self.send_count = 0
        self.max_updates_per_step = max_updates_per_step
        self.sample_queue = sample_queue
        self.minibatch_queue = minibatch_queue

        self.start_thread()

    def start_thread(self):
        threading.Thread(target=self.recv_from_env).start()
        threading.Thread(target=self.send_to_update).start()

    def recv_from_env(self):
        while True:
            self.add(*self.sample_queue.get())
            self.step += 1

    def send_to_update(self):
        while True:
            if self.send_count > (self.step - self.init_steps) * self.max_updates_per_step:
                time.sleep(0.1)
            else:
                self.minibatch_queue.put(tuple(self.sample()))
                self.send_count += 1
