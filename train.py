import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from numpy.random import PCG64DXSM, RandomState
import time
from numpy import float32
import crayons
import math


def train(rank, args, shared_model, optimizer):
    ptitle(f"Train Agent: {rank}")
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    USE_GPU = gpu_id >= 0
    rng = RandomState(PCG64DXSM())
    seed = rng.randint(2147483647)
    if USE_GPU:
        dev = torch.device(f"cuda:{gpu_id}")
    else:
        dev = torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    if USE_GPU:
        torch.cuda.manual_seed(args.seed + rank)

    hidden_size = args.hidden_size
    env = create_env(args.env, args)
    if optimizer is None:
        print("no_shared")
        if args.optimizer == "RMSprop":
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, betas=args.betas, eps=args.opt_eps, amsgrad=args.amsgrad
            )

    player = Agent(env, args, args.seed + rank, gpu_id)
    player.numSteps = range(args.num_steps)
    player.rew_clip = args.rew_clip
    player.noise_std = (
        torch.tensor(math.exp(-(math.log(2 * math.pi) * 0.5 + 0.5))).float().to(device=dev)
    )  # Limit noise std limit so never reach negative entropy
    player.noise_var = player.noise_std.square()
    if args.model == "MLP":
        model = A3C_MLP(env.observation_space.shape[0], env.action_space, args)

    if args.model == "CONV":
        model = A3C_CONV(1, env.action_space, args)

    if args.load_rms_data:
        env.load_running_average("rms_data")
        env.set_training_off()
    else:
        env.set_training_on()

    state, info = env.reset(seed=args.seed + rank)
    state = torch.from_numpy(state).to(device=dev)
    model = model.to(device=dev)
    value_coef = torch.tensor(float(args.value_coef), device=dev)
    entropy_coef = torch.tensor(float(args.entropy_coef), device=dev)
    gamma = torch.tensor(float(args.gamma), device=dev)
    tau = torch.tensor(float(args.tau), device=dev)
    half = torch.tensor(float(0.5), device=dev)
    zeros = torch.zeros((), device=dev)
    ones = torch.ones((), device=dev)
    hx_initial = torch.zeros(1, hidden_size, device=dev)
    cx_initial = torch.zeros(1, hidden_size, device=dev)
    model.train()
    updateError = False
    TRAIN = 1
    done = True
    action_train = player.action_train
    try:
        while TRAIN:
            if done:
                cx = cx_initial
                hx = hx_initial
            else:
                cx = cx.data
                hx = hx.data

            model.load_state_dict(shared_model.state_dict())

            state, hx, cx, done, values, log_probs, rewards, entropy_loss = action_train(state, hx, cx, model, env)

            if done:
                if player.eps_flag:
                    R = values[-1]
                    values = values[:-1]
                else:
                    R = zeros
                player.eps_flag = False
                if not args.load_rms_data:
                    updateDone = False
                    while not updateDone:
                        try:
                            env.load_running_average("rms_data")
                            if env.obs_rms.training_done:
                                env.set_training_off()
                                args.load_rms_data = True

                            updateDone = True
                            if updateError:
                                print(
                                    f"{crayons.yellow(f'Train Agent: {rank} -- updated stats succesfully', bold=True)}"
                                )
                                updateError = False

                        except KeyboardInterrupt:
                            TRAIN = 0
                            print(f"{crayons.yellow('KeyboardInterrupt exception is caught', bold=True)}")
                        except:
                            print(f"{crayons.yellow(f'Error on stats update for - Train Agent: {rank}', bold=True)}")
                            updateError = True

                state, info = env.reset(seed=rng.randint(2147483647))
                player.eps_len = 0
                if USE_GPU:
                    state = torch.from_numpy(state).to(device=dev, non_blocking=True)
                else:
                    state = torch.from_numpy(state)
            else:
                value, _, _, _, _ = model(state, hx, cx)
                R = value.data.squeeze()

            values.append(R)
            if rewards:
                len_rew = len(rewards)
                if USE_GPU:
                    rewards = torch.tensor(rewards).to(device=dev, non_blocking=True)
                else:
                    rewards = torch.as_tensor(rewards)
                gae = zeros
                policy_loss = zeros
                value_loss = zeros
                for i in reversed(range(len_rew)):
                    R = R.mul(gamma).add(rewards[i])
                    advantage = R.sub(values[i])
                    value_loss = advantage.square().mul(half).add(value_loss)

                    # Generalized Advantage Estimataion
                    delta_t = values[i + 1].mul(gamma).sub(values[i]).add(rewards[i])
                    gae = gae.mul(gamma.mul(tau)).add(delta_t.data)
                    policy_loss = policy_loss.sub(log_probs[i].mul(gae))

                model.zero_grad(set_to_none=False)
                (value_loss.mul(value_coef).add(policy_loss).add(entropy_loss.mul(entropy_coef))).backward()
                ensure_shared_grads(model, shared_model, USE_GPU)
                optimizer.step()
    except KeyboardInterrupt:
        TRAIN = 0
        print(f"{crayons.yellow('KeyboardInterrupt exception is caught', bold=True)}")
    except OSError as err:
        print(f"{crayons.yellow(f'OS error: {err}', bold=True)}")
    except Exception as err:
        print(f"{crayons.yellow(f'Unexpected {err=}, {type(err)=}', bold=True)}")
    finally:
        print(f"{crayons.yellow(f'train agent {rank} process finished', bold=True)}")
