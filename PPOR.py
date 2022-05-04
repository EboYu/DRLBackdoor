from collections import namedtuple

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from env.schedulingEnv import SchedulingEnv
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
from DQN import DQN
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.nextstates = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.nextstates[:]

class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hiddensize=32,numlayer=2):
        super(ActorCriticLSTM, self).__init__()
        self.activation = torch.relu
        self.fc1 = nn.Linear(state_dim, hiddensize)
        self.flat_dimension = hiddensize
        self.lstm = nn.LSTM(hiddensize, hiddensize, num_layers=numlayer)
        self.fc2 = nn.Linear(hiddensize, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self,x,hidden):
        x = self.activation(self.fc1(x))
        x = x.view(-1, 1, self.flat_dimension)
        x, hidden = self.lstm(x, hidden)
        x = self.activation(self.fc2(x))
        action_prob = F.softmax(self.actor(x).squeeze(), dim=0)
        value = self.critic(x)
        return action_prob, value, hidden

    def act(self, x,hidden):
        x = self.activation(self.fc1(x))
        x = x.view(-1, 1, self.flat_dimension)
        x, hidden = self.lstm(x, hidden)
        x = self.activation(self.fc2(x))
        action_prob = F.softmax(self.actor(x).squeeze(), dim=0)
        dist = Categorical(action_prob)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), hidden

    def evaluate(self, state, action,hidden):
        x = self.activation(self.fc1(state))
        x = x.view(-1, 1, self.flat_dimension)
        x, hidden = self.lstm(x, hidden)
        x = self.activation(self.fc2(x))
        action_prob = F.softmax(self.actor(x).squeeze(), dim=0)
        dist = Categorical(action_prob)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)
        return action_logprobs, state_values, dist_entropy, hidden

class PPO:
    def __init__(self, state_dim, action_dim, lr, numlayer, gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.counter = 0
        self.lstmlayer = numlayer
        self.policy = ActorCriticLSTM(state_dim, action_dim, numlayer= numlayer).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.policy_old = ActorCriticLSTM(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, hidden):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob,hx = self.policy_old.act(state,hidden)

        return action.item(), action_logprob, hx

    def store(self, state, action, reward, action_logprob, done, next_state):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.nextstates.append(next_state)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        hidden = (Variable(torch.zeros(self.lstmlayer, 1, 32).float().to(self.device)), Variable(torch.zeros(self.lstmlayer, 1, 32).float().to(self.device)))
        old_states = torch.FloatTensor(self.buffer.states).to(self.device)
        old_actions = torch.FloatTensor(self.buffer.actions).to(self.device).unsqueeze(1)

        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device).unsqueeze(1)
        total_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy,_,= self.policy.evaluate(old_states,old_actions,hidden)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            total_loss += loss.mean().item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return total_loss/self.K_epochs

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))