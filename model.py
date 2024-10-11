import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import tanh


class A3C_CONV(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(A3C_CONV, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)

        self.lstm = nn.LSTMCell(1600, self.hidden_size)
        num_outputs = action_space.shape[0]
        self.critic_linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.critic_linear2 = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)

        lrelu_gain = nn.init.calculate_gain("leaky_relu", 0.25)
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        nn.init.kaiming_uniform_(self.critic_linear1.weight, a=0.25, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.orthogonal_(self.actor_linear.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.critic_linear2.weight)

        self.train()

    def forward(self, input, hx, cx):
        x = F.leaky_relu(self.conv1(input.unsqueeze(0)), 0.25)
        x = F.leaky_relu(self.conv2(x), 0.25)
        x = F.leaky_relu(self.conv3(x), 0.25)
        x = F.leaky_relu(self.conv4(x), 0.25)

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        x1 = F.leaky_relu(self.critic_linear1(x), 0.25)
        return (self.critic_linear2(x1), tanh(self.actor_linear(x)), hx, cx)

    def getActions(self, input, hx, cx):
        x = F.leaky_relu(self.conv1(input.unsqueeze(0)), 0.25)
        x = F.leaky_relu(self.conv2(x), 0.25)
        x = F.leaky_relu(self.conv3(x), 0.25)
        x = F.leaky_relu(self.conv4(x), 0.25)

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return (None, tanh(self.actor_linear(x)), hx, cx)


class A3C_MLP(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(A3C_MLP, self).__init__()
        self.hidden_size = args.hidden_size
        self.fc1 = nn.Linear(num_inputs, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        num_outputs = action_space.shape[0]
        self.critic_linear1 = nn.Linear(self.hidden_size, self.hidden_size)

        self.critic_linear2 = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.actor_linear2 = nn.Linear(self.hidden_size, num_outputs)

        nn.init.kaiming_uniform_(self.fc1.weight, a=0.25, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.25, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.critic_linear1.weight, a=0.25, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.orthogonal_(self.actor_linear.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.orthogonal_(self.actor_linear2.weight, gain=3.0**0.5)
        nn.init.xavier_uniform_(self.critic_linear2.weight)
        self.train()

    def forward(self, input, hx, cx):
        x = F.leaky_relu(self.fc1(input), 0.25)
        x = F.leaky_relu(self.fc2(x), 0.25)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        x1 = F.leaky_relu(self.critic_linear1(x), 0.25)
        return (self.critic_linear2(x1), tanh(self.actor_linear(x)), self.actor_linear2(x), hx, cx)

    def getActions(self, input, hx, cx):
        x = F.leaky_relu(self.fc1(input), 0.25)
        x = F.leaky_relu(self.fc2(x), 0.25)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return (None, tanh(self.actor_linear(x)), hx, cx)
