import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from numpy import float32
import torch
from torch import from_numpy, randn_like
import math
from collections import deque
import time


ONE = torch.ones(()).float()
TWO = torch.tensor(2.0).float()
HALF = torch.tensor(0.5).float()
PI = torch.tensor(math.pi).float()
TWO_PI = PI.mul(TWO)
LOG_SQRT_2PI = TWO_PI.sqrt().log()
LOG_2PI_DIV2 = TWO_PI.log().mul(HALF)


def softsignSquashed(input):
    r"""
    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`
    Squashes value to range (0,1)
    """
    return input.mul(HALF).div(input.abs().add(ONE)).add(HALF)


class Agent(object):
    def __init__(self, env, args, seed, gpu_id=-1):
        self.env = env
        self.eps_len = 0
        self.args = args
        self.USE_GPU = gpu_id >= 0
        self.gpu_id = gpu_id
        self.hidden_size = args.hidden_size
        self.eps_flag = False
        torch.manual_seed(seed)
        self.cdev = torch.device("cpu")
        if self.USE_GPU:
            self.dev = torch.device(f"cuda:{gpu_id}")
        else:
            self.dev = torch.device("cpu")
        self.ZEROS = torch.zeros((), device=self.dev).float()
        self.hx_initial = torch.zeros(1, self.hidden_size, device=self.dev).float()
        self.cx_initial = torch.zeros(1, self.hidden_size, device=self.dev).float()
        self.entropy_loss = self.ZEROS
        self.numSteps = None
        self.rew_clip = None
        self.noise_std = None
        self.noise_var = None

    def action_train(self, state, hx, cx, model, env):
        values, log_probs, rewards = [], [], []
        entropy_loss = self.ZEROS
        for step in self.numSteps:
            value, mu, sigma, hx, cx = model(state, hx, cx)
            sigma_var = softsignSquashed(sigma).add(self.noise_var)
            eps = randn_like(mu)
            sigma_std = sigma_var.sqrt()
            action = mu.add(sigma_std.mul(eps)).data
            log_prob = (
                action.sub(mu).square().neg().div(sigma_var.mul(TWO)).sub(sigma_std.log()).sub(LOG_SQRT_2PI).sum()
            )
            entropy = sigma_std.log().add(HALF).add(LOG_2PI_DIV2).sum()
            state, reward, done, truncated, info = env.step(action.clamp(-1.0, 1.0).tolist()[0])
            reward = max(reward * 1.5, self.rew_clip)
            if self.USE_GPU:
                state = from_numpy(state).to(device=self.dev, non_blocking=True)
            else:
                state = from_numpy(state)

            self.eps_len += 1
            if done:
                if truncated or reward <= 0:
                    reward = self.rew_clip
                    values.append(value.squeeze())
                    rewards.append(float32(reward))
                    log_probs.append(log_prob)
                    entropy_loss = entropy_loss.sub(entropy)
                else:
                    self.eps_flag = True
                    values.append(value.data.squeeze())
                return state, hx, cx, done, values, log_probs, rewards, entropy_loss

            values.append(value.squeeze())
            rewards.append(float32(reward))
            log_probs.append(log_prob)
            entropy_loss = entropy_loss.sub(entropy)
        return state, hx, cx, done, values, log_probs, rewards, entropy_loss

    def action_test(self, state, model, env):
        done = False
        reward_sum = 0
        cx = self.cx_initial
        hx = self.hx_initial
        while not done:
            with torch.no_grad():
                value, mu, hx, cx = model.getActions(state, hx, cx)
            state, reward, done, truncated, info = env.step(mu.tolist()[0])
            if self.USE_GPU:
                state = torch.from_numpy(state).to(device=self.dev)
            else:
                state = torch.from_numpy(state)

            self.eps_len += 1
            reward_sum += reward
        return reward_sum, done


class Evaluator(object):
    def __init__(self, max_episode_steps, env, log, args, gpu_id=-1):
        self.testNumber = 0
        self.shortTestNum = 0
        self.log = log
        self.args = args
        self.USE_GPU = gpu_id >= 0
        self.gpu_id = gpu_id
        self.test_model_success = False
        self.loadNew = True
        self.breakTime = 5
        self.score_list = deque(maxlen=100)
        self.maxSteps = float32(5 * max_episode_steps)
        self.env = env
        self.start_time = time.time()

    def runningStatsUpdate(self, model):
        if self.args.model_300_check and self.args.load_rms_data and time.time() - self.start_time > 600:
            self.breakTime = 0
            self.testUpdate(model)

        elif not self.args.load_rms_data:
            if (
                np.mean(self.score_list) > 0
                and self.env.obs_rms.count > self.maxSteps * 5
                and time.time() - self.start_time >= 600
            ):
                self.env.set_training_off()
                self.args.load_rms_data = True
            elif (
                np.mean(self.score_list) < 0 or time.time() - self.start_time < 300
            ) and self.env.obs_rms.count > self.maxSteps:
                self.env.obs_rms.count = self.maxSteps
            self.env.save_running_average("rms_data")

    def testUpdate(self, model):
        if self.testNumber > 0:
            self.full_100eps_test(model)
        elif self.testNumber == 0:
            self.short_test()

    def short_test(self):
        self.loadNew = False
        self.shortTestNum += 1
        temp_score_list = list(self.score_list)[-self.shortTestNum :]
        if self.shortTestNum == 20:
            if np.mean(temp_score_list) > 310:
                self.log.info("***********************************************************************************")
                self.log.info("***********************************************************************************")
                self.log.info("Start 100 Episodes Test! **********************************************************")
                self.testNumber = 1
            else:
                self.shortTestNum = 0
                self.loadNew = True
        elif 0 < self.shortTestNum < 11 and min(temp_score_list) < 300:
            self.shortTestNum = 0
            self.loadNew = True
        elif (
            0 < self.shortTestNum < 20
            and sum(temp_score_list) + max(temp_score_list) * (20 - self.shortTestNum) < 310 * 20
        ):
            self.shortTestNum = 0
            self.loadNew = True

    def full_100eps_test(self, model):
        self.loadNew = False
        if self.testNumber == 100:
            self.log.info(
                f"Finished 100 Episode Test! Reward Mean: {np.mean(self.score_list):.2f} ************************************"
            )
            self.log.info("***********************************************************************************")
            self.log.info("***********************************************************************************")
            if np.mean(self.score_list) > 300:
                self.test_model_success = True
                if self.USE_GPU:
                    with torch.cuda.device(self.gpu_id):
                        state_to_save = model.state_dict()
                        torch.save(state_to_save, f"{self.args.save_model_dir}{self.args.env}.dat")
                else:
                    state_to_save = model.state_dict()
                    torch.save(state_to_save, f"{self.args.save_model_dir}{self.args.env}.dat")

            self.shortTestNum = 0
            self.testNumber = 0
            self.loadNew = True
        else:
            self.testNumber += 1
