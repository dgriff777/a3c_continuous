import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from model import A3C_CONV, A3C_MLP
from player_util import Agent, Evaluator
from numpy.random import PCG64DXSM, RandomState
import time
from collections import deque
from utils import setup_spdlogger
import crayons


def test(args, shared_model):
    ptitle("Test Agent")
    gpu_id = args.gpu_ids[-1]
    USE_GPU = gpu_id >= 0
    if USE_GPU:
        dev = torch.device(f"cuda:{gpu_id}")
    else:
        dev = torch.device("cpu")

    rng = RandomState(PCG64DXSM())
    seed = rng.randint(2147483647)
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed(seed)

    log = setup_spdlogger(args)
    d_args = vars(args)
    for k in d_args.keys():
        log.info(f"{k}: {d_args[k]}")

    env = create_env(args.env, args)
    reward_sum = 0
    num_tests = 0
    player = Agent(env, args, args.seed, gpu_id)
    if args.model == "MLP":
        model = A3C_MLP(env.observation_space.shape[0], env.action_space, args)

    if args.model == "CONV":
        model = A3C_CONV(1, env.action_space, args)

    if args.tensorboard_logger:
        from torch.utils.tensorboard import SummaryWriter

        if args.model == "CONV":
            dummy_input = (torch.zeros(1, 24), torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size))

        if args.model == "MLP":
            dummy_input = (torch.zeros(1, 24), torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size))

        writer = SummaryWriter(f"runs/{args.env}_training")
        writer.add_graph(model, dummy_input, False)
        writer.close()

    if args.load_rms_data:
        env.load_running_average("rms_data")
        env.set_training_off()
    else:
        env.reset_running_stats("rms_data")
        env.set_training_on()

    state, info = env.reset(seed=args.seed + seed)
    state = torch.from_numpy(state).to(device=dev)
    model = model.to(device=dev)
    model.eval()
    playerEval = Evaluator(env.env._max_episode_steps, env, log, args, gpu_id)
    TEST = 1
    done = True
    action_test = player.action_test
    try:
        while TEST:
            if done and playerEval.loadNew:
                model.load_state_dict(shared_model.state_dict())

            reward_sum, done = action_test(state, model, playerEval.env)

            if done:
                num_tests += 1
                playerEval.score_list.append(reward_sum)
                playerEval.log.info(
                    f'Time: {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - playerEval.start_time))}, episode_reward: {reward_sum:.2f}, episode_length: {player.eps_len}, reward_mean: {np.mean(playerEval.score_list):.2f}'
                )
                if args.tensorboard_logger:
                    writer.add_scalar(f"{args.env}_Episode_Rewards", round(reward_sum, 2), num_tests)
                    for name, weight in model.named_parameters():
                        writer.add_histogram(name, weight, num_tests)

                if not playerEval.test_model_success and playerEval.shortTestNum == 0 and playerEval.testNumber == 0:
                    if USE_GPU:
                        with torch.cuda.device(gpu_id):
                            state_to_save = model.state_dict()
                            torch.save(state_to_save, f"{args.save_model_dir}{args.env}.dat")
                    else:
                        state_to_save = model.state_dict()
                        torch.save(state_to_save, f"{args.save_model_dir}{args.env}.dat")

                player.eps_len = 0
                playerEval.runningStatsUpdate(model)
                if args.stop_when_solved and playerEval.test_model_success:
                    TEST = 0

                state, info = playerEval.env.reset(seed=rng.randint(2147483647))
                state = torch.from_numpy(state).to(device=dev)
                time.sleep(playerEval.breakTime)

    except KeyboardInterrupt:
        time.sleep(0.01)
        TEST = 0
        print(f"{crayons.yellow('KeyboardInterrupt exception is caught', bold=True)}")
    except OSError as err:
        print(f"{crayons.yellow(f'OS error: {err}', bold=True)}")
    except Exception as err:
        print(f"{crayons.yellow(f'Unexpected {err=}, {type(err)=}', bold=True)}")
    finally:
        print(f"{crayons.yellow('test agent process finished', bold=True)}")
        if args.tensorboard_logger:
            writer.close()
        playerEval.log.close()
