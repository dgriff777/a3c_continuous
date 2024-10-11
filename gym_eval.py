import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import torch
from environment import create_env
from model import A3C_CONV, A3C_MLP
from player_util import Agent
import gym
import time
from utils import set_console_spdlog


parser = argparse.ArgumentParser(description="A3C_EVAL")
parser.add_argument("--env", default="BipedalWalker-v3", help="environment to train on (default: BipedalWalker-v3)")
parser.add_argument("--num-episodes", type=int, default=100, help="how many episodes in evaluation (default: 100)")
parser.add_argument("--load-model-dir", default="trained_models/", help="folder to load trained models from")
parser.add_argument("--log-dir", default="logs/", help="folder to save logs")
parser.add_argument("--render", action="store_true", help="Watch game as it being played")
parser.add_argument("--model", default="MLP", choices=["MLP", "CONV"], help="Model type to use")
parser.add_argument("--seed", type=int, default=5, help="random seed (default: 5)")
parser.add_argument("--gpu-id", type=int, default=-1, help="GPU to use [-1 CPU only] (default: -1)")
parser.add_argument("--hidden-size", type=int, default=160, help="LSTM Cell hidden size (default: 160)")
args = parser.parse_args()

torch.manual_seed(args.seed)
saved_state = torch.load(
    f"{args.load_model_dir}{args.env}.dat", map_location=lambda storage, loc: storage, weights_only=True
)

log = set_console_spdlog(args)
gpu_id = args.gpu_id
d_args = vars(args)
for k in d_args.keys():
    log.info(f"{k}: {d_args[k]}")

if gpu_id >= 0:
    dev = torch.device(f"cuda:{gpu_id}")
else:
    dev = torch.device("cpu")

env = create_env(args.env, args)
num_tests = 0
reward_total_sum = 0
player = Agent(env, args, gpu_id)
if args.model == "MLP":
    model = A3C_MLP(env.observation_space.shape[0], env.action_space, args)
elif args.model == "CONV":
    model = A3C_CONV(args.stack_frames, env.action_space, args)

model = model.to(device=dev)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        model.load_state_dict(saved_state)
else:
    model.load_state_dict(saved_state)

model.eval()
start_time = time.time()
env.load_running_average("rms_data")
env.set_training_off()
if args.seed:
    state, info = env.reset(seed=args.seed)
else:
    state, info = env.reset()

action_test = player.action_test
done = True
try:
    for i_episode in range(args.num_episodes):
        player.eps_len = 0
        state = torch.from_numpy(state).to(device=dev)
        reward_sum, done = action_test(state, model, env)
        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log.info(
                f"Time: {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, episode_reward: {reward_sum:.2f}, episode_length: {player.eps_len}, reward_mean: {reward_mean:.2f}"
            )
            state, info = env.reset()

except KeyboardInterrupt:
    print("KeyboardInterrupt exception is caught")
finally:
    print("gym evalualtion process finished")
    env.close()
    log.close()
