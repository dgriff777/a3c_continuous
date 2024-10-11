import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torch.optim import Optimizer
from collections import defaultdict
from math import sqrt
import time
import torch.nn.functional as F


CONST_1 = torch.ones(()).float()
CONST_2 = torch.ones(()).float() * 2.0


class SharedRMSprop(Optimizer):
    """Implements RMSprop algorithm with shared states."""

    def __init__(self, params, lr=7e-4, alpha=0.99, eps=0.1, weight_decay=0, momentum=0, centered=False):
        defaults = defaultdict(
            lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered
        )
        super(SharedRMSprop, self).__init__(params, defaults)

        self.ONE = torch.ones(())
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(())
                state["grad_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["square_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["momentum_buffer"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["square_avg"].share_memory_()
                state["step"].share_memory_()
                state["grad_avg"].share_memory_()
                state["momentum_buffer"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"].add_(self.ONE)

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group["eps"])
                else:
                    avg = square_avg.sqrt().add_(group["eps"])

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    # Need to avoid version tracking for parameter.
                    p.data.add_(buf, alpha=-group["lr"])
                else:
                    # Need to avoid version tracking for parameter.
                    p.data.addcdiv_(grad, avg, value=-group["lr"])

        return loss


class SharedAdam(Optimizer):
    """Implements Adam algorithm with shared states."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-3, weight_decay=0, amsgrad=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)
        defaults["torch_eps"] = torch.tensor(eps).float()
        defaults["beta1"], defaults["beta2"] = betas
        defaults["beta2T"] = torch.tensor(defaults["beta2"]).float()
        defaults["stepNum"] = torch.zeros(()).float()
        defaults["OneMinusBeta1"] = CONST_1.sub(defaults["beta1"]).float()
        defaults["OneMinusBeta2"] = CONST_1.sub(defaults["beta2T"]).float()
        defaults["NEG_LR"] = -lr
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["max_exp_avg_sq"] = torch.zeros_like(p) + defaults["torch_eps"].square()

    def share_memory(self):
        self.defaults["stepNum"].share_memory_()
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["max_exp_avg_sq"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stepFlag = 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if stepFlag:
                    defaults = self.defaults
                    amsgrad = defaults["amsgrad"]
                    OneMinusBeta1 = defaults["OneMinusBeta1"]
                    OneMinusBeta2 = defaults["OneMinusBeta2"]
                    beta2T = defaults["beta2T"]
                    defaults["stepNum"].add_(CONST_1)
                    step_t = defaults["stepNum"].item()
                    bias_correction1 = 1 - defaults["beta1"] ** step_t
                    bias_correction2 = 1 - defaults["beta2"] ** step_t
                    bias_correction2_sqrt = sqrt(bias_correction2)
                    step_size_neg = defaults["NEG_LR"] * bias_correction2_sqrt / bias_correction1
                    stepFlag = 0

                grad = p.grad
                state = self.state[p]

                exp_avg = state["exp_avg"]
                exp_avg.lerp_(grad, OneMinusBeta1)
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2T).addcmul_(grad, grad, value=OneMinusBeta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running avg till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the mean of 2nd moment running avg and maximum of all 2nd moment running avg till now for normalizing running avg of gradient
                    denom = exp_avg_sq.add(max_exp_avg_sq).div(CONST_2).sqrt()
                else:
                    denom = exp_avg_sq.sqrt().add(defaults["torch_eps"])

                p.data.addcdiv_(exp_avg, denom, value=step_size_neg)
        return loss
