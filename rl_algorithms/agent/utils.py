import torch
import numpy as np
import gym
import os
from collections import deque
import random
import torch.multiprocessing as mp

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed, env):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    env.seed(seed)
    env.action_space.seed(seed)

def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def random_augment(images, rad_height, rad_width):
    n, c, h, w = images.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_images = torch.empty((n, c, _h, _w), device=images.device).float()
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped_images[i][:] = image[:, h11:h11 + _h, w11:w11 + _w]
    return cropped_images

def evaluate(env, agent, num_episodes, L, step, args):
    for i in range(num_episodes):
        obs = env.reset()
        #video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with eval_mode(agent):
                obs = obs[:, args.rad_offset: args.image_size + args.rad_offset, args.rad_offset: args.image_size + args.rad_offset]
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            #video.record(env)
            episode_reward += reward

        #video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


class BufferQueue(object):
    """Queue to transfer arbitrary number of data between processes"""
    def __init__(self, num_items, max_size=10, start_method='spawn'):
        self.max_size = max_size
        ctx = mp.get_context(start_method)
        self.queues = [ctx.Queue(max_size) for _ in range(num_items)]

    def put(self, *items):
            for queue, item in zip(self.queues, items):
                queue.put(item)

    def get(self):
        return [queue.get() for queue in self.queues]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
