import numpy as np
np.bool8 = np.bool_
import gym
from numpy import float32, sqrt, square
from gym import spaces, Wrapper
import pickle


def create_env(env_id, args):
    hardcore = False
    render = "rgb_array"
    if env_id == "BipedalWalkerHardcore-v3":
        env_id = "BipedalWalker-v3"
        hardcore = True
    if hasattr(args, "render"):
        if args.render:
            render = "human"
    env = gym.make(env_id, hardcore=hardcore, render_mode=render)
    env = BipedalEnv(env, args)
    return env


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, float32)
        self.var = np.ones(shape, float32)
        self.count = float32(epsilon)
        self.training_done = False

    def update(self, arr):
        batch_mean = arr
        batch_count = 1
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_2 = self.var * self.count + square(delta) * self.count * batch_count / tot_count
        self.var = m_2 / tot_count
        self.count = tot_count


class BipedalEnv(Wrapper):
    def __init__(self, env, args):
        super(BipedalEnv, self).__init__(env)
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        self.is_training = True
        self.envid = args.env

    def reset_running_stats(self, path):
        self.obs_rms = RunningMeanStd(shape=self.env.observation_space.shape)
        self.save_running_average(path)

    def set_training_on(self):
        self.is_training = True
        self.obs_rms.training_done = False

    def set_training_off(self):
        self.obs_rms.std = sqrt(self.obs_rms.var.clip(1e-4, None))
        self.is_training = False
        self.obs_rms.training_done = True

    def obs_norm(self, obs):
        if self.is_training:
            self.obs_rms.update(obs)
            return ((obs - self.obs_rms.mean) / sqrt(self.obs_rms.var.clip(1e-4, None))).clip(-20.0, 20.0)
        else:
            return ((obs - self.obs_rms.mean) / self.obs_rms.std).clip(-10.0, 10.0)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        return self.obs_norm(ob).reshape(1, 24).astype(float32), info

    def step(self, action):
        ob, rew, done, truncated, info = self.env.step(action)
        if truncated:
            done = True
        return self.obs_norm(ob).reshape(1, 24).astype(float32), rew, done, truncated, info

    def save_running_average(self, path):
        for rms, name in zip([self.obs_rms], ["obs_rms"]):
            with open(f"{path}/{name}_{self.envid}.pkl", "wb") as file_handler:
                pickle.dump(rms, file_handler, protocol=-1)

    def load_running_average(self, path):
        for name in ["obs_rms"]:
            with open(f"{path}/{name}_{self.envid}.pkl", "rb") as file_handler:
                setattr(self, name, pickle.load(file_handler))
