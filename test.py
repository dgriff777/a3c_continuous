from __future__ import division
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import gym


def test(args, shared_model):
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    env = create_env(args.env, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    if args.model == 'MLP':
        player.model = A3C_MLP(
            player.env.observation_space.shape[0], player.env.action_space)
    if args.model == 'CONV':
        player.model = A3C_CONV(args.stack_frames, player.env.action_space)
    player.state = player.env.reset()

    player.state = torch.from_numpy(player.state).float()
    player.model.eval()
    while True:
        if player.done:
            player.model.load_state_dict(shared_model.state_dict())

        state, reward, player.done, player.info = player.env.step(
            player.action_test())
        player.state = torch.from_numpy(state).float()
        player.eps_len += 1
        player.done = player.done or player.eps_len >= player.args.max_episode_length
        reward_sum += reward

        if player.done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if reward_sum > args.save_score_level:
                player.model.load_state_dict(shared_model.state_dict())
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                    args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            time.sleep(60)
            player.state = torch.from_numpy(state).float()
