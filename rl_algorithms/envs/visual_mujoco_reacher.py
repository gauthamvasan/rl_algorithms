import cv2
import gym

import numpy as np

from collections import deque
from gym.spaces import Box


class VisualMujocoReacher2D(gym.Wrapper):
    def __init__(self, tol, image_shape, image_period):
        """

        Args:
            tol (float): Smaller the value, smaller the target size (e.g., 0.009, 0.018, 0.036, 0.072, etc.)
            image_shape: A list/tuple with 3 integers
            image_period (int): Update image obs only every 'image_period' steps
        """
        super().__init__(gym.make('Reacher-v2').unwrapped)
        self._tol = tol
        self._image_period = image_period
        low = list(self.env.observation_space.low[0:4]) + list(self.env.observation_space.low[6:8])
        high = list(self.env.observation_space.high[0:4]) + list(self.env.observation_space.high[6:8])
        self.proprioception_space = Box(np.array(low), np.array(high))

        self._use_image = False
        if image_shape != (0, 0, 0):
            self._image_buffer = deque([], maxlen=image_shape[0] // 3)
            self._use_image = True
            print('time period:', image_period)

        self.image_space = Box(low=0, high=255, shape=image_shape)

        # remember to reset
        self._latest_image = None
        self._reset = False
        self._step = 0

    def step(self, a):
        assert self._reset

        ob, _, done, info = self.env.step(a)
        ob = self._get_ob(ob)
        self._step += 1

        dist_to_target = -info["reward_dist"]

        reward = -1
        if dist_to_target <= self._tol:
            info['reached'] = True
            done = True

        if self._use_image and (self._step % self._image_period) == 0:
            new_img = self._get_new_img()
            self._image_buffer.append(new_img)
            self._latest_image = np.concatenate(self._image_buffer, axis=0)

        if done:
            self._reset = False

        return self._latest_image, ob, reward, done, info

    def reset(self):
        ob = self.env.reset()
        ob = self._get_ob(ob)

        if self._use_image:
            new_img = self._get_new_img()
            for _ in range(self._image_buffer.maxlen):
                self._image_buffer.append(new_img)

            self._latest_image = np.concatenate(self._image_buffer, axis=0)

        self._reset = True
        self._step = 0
        return self._latest_image, ob

    def _get_new_img(self):
        img = self.env.render(mode='rgb_array')
        img = img[150:400, 50:450, :]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = np.transpose(img, [2, 0, 1])  # c, h, w

        return img

    def _get_ob(self, ob):

        return np.array(list(ob[0:4]) + list(ob[6:8]))

    def close(self):
        super().close()

        del self


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    print(torch.__version__)
    env = VisualMujocoReacher2D(0.072, (9, 125, 200), image_period=3)
    img, ob = env.reset()
    img = np.transpose(img, [1, 2, 0])
    # create two subplots
    plt.ion()
    ax1 = plt.subplot(1, 1, 1)
    im1 = ax1.imshow(img[:, :, 6:9])

    waitKey = 1
    while True:
        im1.set_data(img[:, :, 6:9])
        plt.pause(0.05)
        a = env.action_space.sample()
        img, ob, reward, done, info = env.step(a)
        print(ob)
        img = np.transpose(img, [1, 2, 0])
        if done:
            env.reset()
    plt.show()
