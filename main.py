import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import multiprocessing as mp
from environment import create_env
from model import A3C_MLP, A3C_CONV
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description="A3C")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)")
parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy loss coefficient (default: 0.01)")
parser.add_argument("--opt-eps", type=float, default=1e-8, help="adam eps parameter (default: 1e-8)")
parser.add_argument("--value-coef", type=float, default=0.4, help="value loss coefficient (default: 0.4)")
parser.add_argument("--gamma", type=float, default=0.98, help="discount factor for rewards (default: 0.98)")
parser.add_argument("--tau", type=float, default=0.8, help="parameter for GAE (default: 0.8)")
parser.add_argument("--load-rms-data", action="store_true", help="load training data")
parser.add_argument("--seed", type=int, default=5, help="random seed (default: 5)")
parser.add_argument("--rew-clip", type=float, default=-20.0, help="min reward clip value (default: -20.0)")
parser.add_argument("--workers", type=int, default=18, help="# training agents to use (default: 18)")
parser.add_argument("--num-steps", type=int, default=20, help="Number of forward steps (default: 20)")
parser.add_argument("--env", default="BipedalWalker-v3", help="Gym environment (default: BipedalWalker-v3)")
parser.add_argument("--shared-optimizer", action="store_true", help="use an optimizer with shared stats.")
parser.add_argument("--load", action="store_true", help="load a trained model")
parser.add_argument("--optimizer", default="Adam", choices=["Adam", "RMSprop"], help="Adam/RMSprop optimizer")
parser.add_argument("--load-model-dir", default="trained_models/", help="Trained models folder")
parser.add_argument("--save-model-dir", default="trained_models/", help="folder to save trained models")
parser.add_argument("--log-dir", default="logs/", help="folder to save logs")
parser.add_argument("--model", default="MLP", choices=["MLP", "CONV"], help="Model type to use")
parser.add_argument("--gpu-ids", type=int, default=[-1], nargs="+", help="GPUs to use [-1 CPU only] (default: -1)")
parser.add_argument("--amsgrad", action="store_true", help="Adam optimizer amsgrad parameter")
parser.add_argument("--hidden-size", type=int, default=160, help="LSTM Cell hidden size (default: 160)")
parser.add_argument("--tensorboard-logger", action="store_true", help="Creates tensorboard logger")
parser.add_argument("--stop-when-solved", action="store_true", help="stop training after successful 100 episode test")
parser.add_argument("--model-300-check", action="store_true", help="Actively test for a successful model")
parser.add_argument("--betas", type=float, default=[0.9, 0.999], nargs="+", help="Adam betas (default: [0.9, 0.999])")


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    #    mp.set_start_method("spawn")
    env = create_env(args.env, args)
    if args.model == "MLP":
        shared_model = A3C_MLP(env.observation_space.shape[0], env.action_space, args)

    if args.model == "CONV":
        shared_model = A3C_CONV(1, env.action_space, args)

    if args.load:
        saved_state = torch.load(f"{args.load_model_dir}{args.env}.dat", map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)

    shared_model.share_memory()
    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, betas=args.betas, eps=args.opt_eps, amsgrad=args.amsgrad
            )
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []
    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    time.sleep(0.01)
    processes.append(p)
    n = 0
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        time.sleep(0.01)
        processes.append(p)
    for p in processes:
        if n < 1:
            p.join()
            n += 1
        else:
            p.terminate()
