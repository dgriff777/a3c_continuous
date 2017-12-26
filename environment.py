from __future__ import division
import gym
import numpy as np
from collections import deque
from gym import spaces


def create_env(env_id, args):
    env = gym.make(env_id)
    env = frame_stack(env, args)
    return env


class frame_stack(gym.Wrapper):
    def __init__(self, env, args):
        super(frame_stack, self).__init__(env)
        self.stack_frames = args.stack_frames
        self.frames = deque([], maxlen=self.stack_frames)
        self.obs_norm = MaxMinFilter()

    def _reset(self):
        ob = self.env.reset()
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.float32(ob)
        ob = self.obs_norm(ob)
        self.frames.append(ob)
        return self._observation(), rew, done, info

    def _observation(self):
        assert len(self.frames) == self.stack_frames
        return np.stack(self.frames, axis=0)


class MaxMinFilter():
    def __init__(self):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 10.0
        self.new_mind = -10.0

    def __call__(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        return new_obs
