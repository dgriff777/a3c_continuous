from __future__ import division
import os

os.environ["OMP_NUM_THREADS"] = "1"
from math import pi as PI
import numpy as np
from numpy import fromiter, float32
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normal  # , pi


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size

    def action_train(self):
        value, mu, sigma, self.hx, self.cx = self.model(
            self.state.unsqueeze(0), self.hx, self.cx
        )
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        pi = np.array([PI])
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = torch.randn(mu.size()).float().cuda()
                pi = torch.from_numpy(pi).float().cuda()
        else:
            eps = torch.randn(mu.size()).float()
            pi = torch.from_numpy(pi).float()

        action = (mu + sigma.sqrt() * eps).data
        act = action
        prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(
            fromiter(action.tolist()[0], dtype=float32)
        )  # faster than action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = torch.zeros(1, self.hidden_size).cuda()
                        self.hx = torch.zeros(1, self.hidden_size).cuda()
                else:
                    self.cx = torch.zeros(1, self.hidden_size)
                    self.hx = torch.zeros(1, self.hidden_size)

            value, mu, sigma, self.hx, self.cx = self.model(
                self.state.unsqueeze(0), self.hx, self.cx
            )
            mu = torch.clamp(mu, -1.0, 1.0)
        state, self.reward, self.done, self.info = self.env.step(
            fromiter(mu.tolist()[0], dtype=float32)
        )
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
