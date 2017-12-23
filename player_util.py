from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normal, pi


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


    def action_train(self):
        if self.done:
            self.cx = Variable(torch.zeros(1, 128), volatile=False)
            self.hx = Variable(torch.zeros(1, 128), volatile=False)
        else:
            self.cx = Variable(self.cx.data, volatile=False)
            self.hx = Variable(self.hx.data, volatile=False)
        value, mu, sigma, (self.hx, self.cx) = self.model((self.state, (self.hx, self.cx)))
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        action = (mu + sigma.sqrt()*Variable(torch.randn(mu.size()))).data

        prob = normal(action, mu, sigma)
        action = torch.clamp(action, -1.0, 1.0)

        entropy = 0.5*((sigma * 2*pi.expand_as(sigma)).log()+1)
        self.entropies.append(entropy)
        log_prob=(prob + 1e-6).log()
        self.log_probs.append(log_prob)
        if np.isnan(action.numpy()).any():
            print 'Nan in list'
        state, reward, self.done, self.info = self.env.step(action.numpy()[0])
        reward = max(min(reward, 1.0), -1.0)
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        if self.done:
            self.cx = Variable(torch.zeros(1, 128), volatile=True)
            self.hx = Variable(torch.zeros(1, 128), volatile=True)
        else:
            self.cx = Variable(self.cx.data, volatile=True)
            self.hx = Variable(self.hx.data, volatile=True)

        value, mu, sigma, (self.hx, self.cx) = self.model((self.state, (self.hx, self.cx)))
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu
        return action.numpy()[0]


    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
