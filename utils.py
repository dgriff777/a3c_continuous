import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import json
import spdlog as spd
import math


CPU_DEV = torch.device("cpu")


def setup_spdlogger(args):
    log = spd.FileLogger(
        f"fast_logger_{args.env}", f"{args.log_dir}/{args.env}_log", multithreaded=False, truncate=True
    )
    log.set_pattern("%H:%M:%S.%f: %v")
    log.set_level(spd.LogLevel.INFO)
    log.flush_on(spd.LogLevel.INFO)
    return log


def set_console_spdlog(args):
    log = spd.ConsoleLogger(f"fast_logger_{args.env}_mon", False, True, True)
    log.set_pattern("%H:%M:%S.%f: %v")
    log.set_level(spd.LogLevel.INFO)
    log.flush_on(spd.LogLevel.INFO)
    return log


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, "r"))
    return json_object


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif gpu:
            shared_param._grad = param.grad.to(device=CPU_DEV, non_blocking=True)
            if not shared_param.grad.bool().any():
                shared_param._grad = param.grad.to(device=CPU_DEV)
        else:
            shared_param._grad = param.grad
