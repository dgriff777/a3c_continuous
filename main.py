from __future__ import print_function, division
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import A3C_MLP, A3C_CONV
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description="A3C")
parser.add_argument(
    "-l", "--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "-ec",
    "--entropy-coef",
    type=float,
    default=0.01,
    help="entropy loss coefficient (default: 0.01)",
)
parser.add_argument(
    "-vc",
    "--value-coef",
    type=float,
    default=0.5,
    help="value loss coefficient (default: 0.5)",
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for rewards (default: 0.99)",
)
parser.add_argument(
    "-t", "--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)"
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=32,
    help="how many training processes to use (default: 32)",
)
parser.add_argument(
    "-ns",
    "--num-steps",
    type=int,
    default=20,
    help="number of forward steps in A3C (default: 20)",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=10000,
    help="maximum length of an episode (default: 10000)",
)
parser.add_argument(
    "-ev",
    "--env",
    default="BipedalWalker-v2",
    help="environment to train on (default: BipedalWalker-v2)",
)
parser.add_argument(
    "-so",
    "--shared-optimizer",
    default=True,
    help="use an optimizer with shared statistics.",
)
parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
parser.add_argument(
    "-sm",
    "--save-max",
    action="store_true",
    help="Save model on every test run high score matched or bested",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="Adam",
    choices=["Adam", "RMSprop"],
    help="optimizer choice of Adam or RMSprop",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-smd",
    "--save-model-dir",
    default="trained_models/",
    help="folder to save trained models",
)
parser.add_argument("-lg", "--log-dir", default="logs/", help="folder to save logs")
parser.add_argument(
    "-m", "--model", default="MLP", choices=["MLP", "CONV"], help="Model type to use"
)
parser.add_argument(
    "-sf",
    "--stack-frames",
    type=int,
    default=1,
    help="Choose number of observations to stack",
)
parser.add_argument(
    "-gp",
    "--gpu-ids",
    type=int,
    default=[-1],
    nargs="+",
    help="GPUs to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-a", "--amsgrad", action="store_true", help="Adam optimizer amsgrad parameter"
)
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=128,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-tl",
    "--tensorboard-logger",
    action="store_true",
    help="Creates tensorboard logger to see graph of model, view model weights and biases, and monitor test agent reward progress",
)

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")
    env = create_env(args.env, args)
    if args.model == "MLP":
        shared_model = A3C_MLP(env.observation_space.shape[0], env.action_space, args)
    if args.model == "CONV":
        shared_model = A3C_CONV(args.stack_frames, env.action_space, args)
    if args.load:
        saved_state = torch.load(
            f"{args.load_model_dir}{args.env}.dat",
            map_location=lambda storage, loc: storage,
        )
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad
            )
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    time.sleep(0.001)
    processes.append(p)
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        time.sleep(0.001)
        processes.append(p)
    for p in processes:
        p.join()
        time.sleep(0.001)
