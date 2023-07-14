import torch
from torch import nn


# Actor network
class Actor(nn.Module):
    def __init__(self, hidden_size, obs_space, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.max_action = torch.Tensor(action_space.high)
        self.num_inputs = obs_space.shape[0]
        self.num_outputs = action_space.shape[0]
        self.network = nn.Sequential(nn.Linear(self.num_inputs, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                     nn.ReLU(),
                                     )
        self.tanh = nn.Tanh()
        self.mu = nn.Linear(hidden_size, self.num_outputs)
      #  self.mu.weight.data.mul_(0.1)
       # self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = self.network(inputs)
        mu = self.mu(x)

        out = self.max_action * self.tanh(mu)
        return out

    def get_action(self, state):
        state = torch.FloatTensor(state) #.to(device)  # .unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()  # [0, 0]


class Critic(nn.Module):
    def __init__(self, hidden_size, obs_space, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.num_inputs = obs_space.shape[0]
        num_outputs = action_space.shape[0]

        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
      #  self.V.weight.data.mul_(0.1)
      #  self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = self.relu(self.ln1(self.fc1(inputs)))

        x = torch.cat([x, actions], 1)
        x = self.relu(self.ln2(self.fc2(x)))
        V = self.V(x)
        return V
