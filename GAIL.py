import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
from env.schedulingEnv import SchedulingEnv
from env.jobGenerator1 import JobGenLoader1
from DQN import DQN
import numpy as np
import random
import pickle,csv
import pandas as pd
from collections import deque

import PPOR

num_experts = 1

class replay_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, done, value):
        observation = np.expand_dims(observation, 0)
        self.memory.append([observation, action, reward, done, value])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, dones, values, returns, advantages = zip(* batch)
        #observations, actions, _, _, _= zip(*batch)
        return np.concatenate(observations, 0), actions, returns, advantages
    def sample1(self, sample_index):
        batch = np.array(self.memory)[sample_index]
        # observations, actions, rewards, dones, values, returns, advantages = zip(* batch)
        observations, actions, _, _, _= zip(* batch)
        return np.concatenate(observations, 0), actions

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[3]) + traj[4]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[2] + Value_previous * self.gamma * (1 - traj[3]) - traj[4]
            Adv = delta + (1 - traj[3]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[4]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

def extractExperts(normalenv):
    norDqn = DQN(SchedulingEnv.action_space_dim,SchedulingEnv.state_space_dim)
    norDqn.load('models/BackdoorDQN_3_10_1000_999_0.pkl')
    advDqn = DQN(SchedulingEnv.action_space_dim,SchedulingEnv.state_space_dim)
    advDqn.load('models/BackdoorDQNA_3_10_1000_986_3.pkl')
    i=0
    rewards =[]
    labels = []
    current_ep_reward = 0
    state_n = normalenv.state
    labels.extend(normalenv.loader.train_labels)
    poison = False
    poisonLen = 0
    states = []
    actions = []
    rewards = []
    while True:
        action_n = norDqn.choose_action(state_n, 0)
        action_a = advDqn.choose_action(state_n, 0)
        if normalenv.pattern_trigger():
            poison = True
            poisonLen = normalenv.loader.poisonlen
        if poisonLen <= 0:
            poison = False
        else:
            poisonLen -= 1
        if poison:
            next_state_n, reward_n, done_n, _ = normalenv.step(action_a)
            actions.append(action_a)
        else:
            next_state_n, reward_n, done_n, _ = normalenv.step(action_n)
            actions.append(action_n)
        current_ep_reward += reward_n
        states.append(state_n)
        rewards.append(reward_n)
        state_n = next_state_n
        if done_n:
            break
    print('Expert reward: ', current_ep_reward)
    normalenv.reset(False)
    return states,actions,rewards

class gail(object):
    def __init__(self, env, episode, capacity, gamma, lam,
                 value_learning_rate, policy_learning_rate, discriminator_learning_rate, batch_size, policy_iter, disc_iter, value_iter, epsilon, entropy_weight, train_iter, clip_grad, render):
        self.env = env
        self.episode = episode
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.value_learning_rate = value_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.batch_size = batch_size
        self.experts = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_iter = policy_iter
        self.disc_iter = disc_iter
        self.value_iter = value_iter
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.train_iter = train_iter
        self.clip_grad = clip_grad
        self.render = render

        self.observation_dim = 13
        self.action_dim = 10
        self.lstmLayer = 2
        self.ppo = PPOR.PPO(self.observation_dim,self.action_dim,self.value_learning_rate,self.lstmLayer,self.gamma,10,0.2,self.device)

        self.discriminator = discriminator(self.observation_dim + self.action_dim)
        self.buffer = replay_buffer(self.capacity, self.gamma, self.lam)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_learning_rate)
        self.disc_loss_func = nn.BCELoss()
        self.weight_reward = None
        self.weight_custom_reward = None

    def randomsample(self, expert_observations, expert_actions):
        bnum = int(len(expert_actions)/self.batch_size)
        idx = np.random.randint(1,bnum)
        expert_observations = np.vstack(expert_observations)
        expert_actions = np.vstack(expert_actions)
        sample_index = np.arange(idx*self.batch_size, (idx+1)*self.batch_size, 1, int)
        batch_state = torch.FloatTensor(expert_observations[sample_index]).to(self.device)
        batch_action = torch.LongTensor(expert_actions[sample_index]).to(self.device)
        return batch_state,batch_action, sample_index


    def discriminator_train(self):
        expert_observations_all, expert_actions_all, expert_rewards_all = self.experts
        expert_observations, expert_actions, sample_index = self.randomsample(expert_observations_all,expert_actions_all)
        expert_actions_index = expert_actions
        expert_actions = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        expert_actions.scatter_(1, expert_actions_index, 1)
        expert_trajs = torch.cat([expert_observations, expert_actions], 1)
        expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0).to(self.device)

        observations, actions = self.buffer.sample1(sample_index)
        observations = torch.FloatTensor(observations).to(self.device)
        actions_index = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        actions_dis = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        actions_dis.scatter_(1, actions_index, 1)
        trajs = torch.cat([observations, actions_dis], 1)
        labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0).to(self.device)
        totalLoss = 0
        for _ in range(self.disc_iter):
            expert_loss = self.disc_loss_func(self.discriminator.forward(expert_trajs), expert_labels)
            current_loss = self.disc_loss_func(self.discriminator.forward(trajs), labels)

            loss = expert_loss + current_loss
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()
            totalLoss+=loss.item()
        return  totalLoss/self.disc_iter

    def get_reward(self, observation, action):
        observation = torch.FloatTensor(np.expand_dims(observation, 0))
        action_tensor = torch.zeros(1, self.action_dim)
        action_tensor[0, action] = 1.
        traj = torch.cat([observation, action_tensor], 1)
        reward = self.discriminator.forward(traj)
        reward = - reward.log()
        return reward.detach().item()

    def train(self):
        for i in range(self.episode):
            obs,_ = self.env.reset()
            self.experts = extractExperts(self.env)
            total_reward = 0
            total_custom_reward = 0
            hidden = (Variable(torch.zeros(self.lstmLayer, 1, 32).float().to(self.device)), Variable(torch.zeros(self.lstmLayer, 1, 32).float().to(self.device)))
            while True:
                action, action_prob,hidden = self.ppo.select_action(obs,hidden)
                next_obs, reward, done, _ = self.env.step(action)
                custom_reward = self.get_reward(obs, action)

                self.ppo.store(obs,action,reward,action_prob,done,next_obs)
                self.buffer.store(obs, action, custom_reward, done, None)
                total_reward += reward
                total_custom_reward += custom_reward
                obs = next_obs
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    if not self.weight_custom_reward:
                        self.weight_custom_reward = total_custom_reward
                    else:
                        self.weight_custom_reward = 0.99 * self.weight_custom_reward + 0.01 * total_custom_reward

                    # self.buffer.process()
                    disLoss = self.discriminator_train()
                    pploss = self.ppo.update()
                    self.buffer.clear()
                    print(
                        'episode: {}  reward: {:.2f}  custom_reward: {:.3f}  weight_reward: {:.2f}  weight_custom_reward: {:.4f} discriminator_loss: {:.3f} ppo_loss: {:.3f}'.format(
                            i + 1, total_reward, total_custom_reward, self.weight_reward, self.weight_custom_reward,
                            disLoss, pploss))

                    break
            if i>1000 and i%1000==0:
                self.ppo.save('model/gail_ppo_{}.pkl'.format(i))

    def run(self):
        for i in range(self.episode):
            obs = self.env.reset()
            self.experts = extractExperts(self.env)
            total_reward = 0
            total_custom_reward = 0
            hidden = (Variable(torch.zeros(1, 1, 32).float()), Variable(torch.zeros(1, 1, 32).float()))
            while True:
                action, action_prob, hidden = self.ppo.select_action(obs, hidden)
                next_obs, reward, done, _ = self.env.step(action)
                custom_reward = self.get_reward(obs, action)

                self.ppo.store(obs, action, reward, action_prob, done, next_obs)
                self.buffer.store(obs, action, custom_reward, done, None)
                total_reward += reward
                total_custom_reward += custom_reward
                obs = next_obs
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    if not self.weight_custom_reward:
                        self.weight_custom_reward = total_custom_reward
                    else:
                        self.weight_custom_reward = 0.99 * self.weight_custom_reward + 0.01 * total_custom_reward

                    # self.buffer.process()
                    disLoss = self.discriminator_train()
                    aloss, vloss = self.ppo.update()
                    self.buffer.clear()
                    print(
                        'episode: {}  reward: {:.2f}  custom_reward: {:.3f}  weight_reward: {:.2f}  weight_custom_reward: {:.4f} discriminator_loss: {:.3f} ppo_value_loss: {:.3f} ppo_actor_loss: {:.3f}'.format(
                            i + 1, total_reward, total_custom_reward, self.weight_reward, self.weight_custom_reward,
                            disLoss, vloss, aloss))

                    break
            if i > 1000 and i % 1000 == 0:
                self.ppo.save('model/gail_ppo_{}.pkl'.format(i))



def train():
    state_dim = 13
    action_dim = 10

    loader = JobGenLoader1(requestRate=10)
    env = SchedulingEnv(loader)

    trainer = gail(
        env=env,
        episode=10000000,
        capacity=1000,
        gamma=0.99,
        lam=0.95,
        value_learning_rate=3e-4,
        policy_learning_rate=3e-4,
        discriminator_learning_rate=3e-4,
        batch_size=128,
        policy_iter=1,
        disc_iter=10,
        value_iter=1,
        epsilon=0.2,
        entropy_weight=1e-4,
        train_iter=500,
        clip_grad=40,
        render=False
    )
    trainer.train()

if __name__ == '__main__':
    train()