import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from env.jobGenerator1 import JobGenLoader1
from env.jobGenerator0 import JobGenLoader0
from torch.autograd import Variable
from env.schedulingEnv import SchedulingEnv
from env.schedulingEnvPartial import SchedulingEnv as SchedulingEnvP
from matplotlib import pyplot as plt
import torch
import numpy

BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
EPISILO = 0.9
EPS_END = 0.0002
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100
NUM_ACTIONS = SchedulingEnv.action_space_dim
NUM_STATES = SchedulingEnv.state_space_dim
ENV_A_SHAPE = 0
hiddensize= 32
numlayer = 2

class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, inputsize, outputsize):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputsize, hiddensize)
        self.fc2 = nn.Linear(hiddensize, 64)
        self.fc3 = nn.Linear(64, outputsize)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 1, hiddensize)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.fc3(x)
        return action_prob

class DQN:
    """docstring for DQN"""

    def __init__(self, actionsize, statesize, cudanum=0):
        super(DQN, self).__init__()
        self.actionSize =actionsize
        self.stateSize = statesize
        self.device = torch.device("cuda:{}".format(cudanum) if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(statesize,actionsize).to(self.device), Net(statesize,actionsize).to(self.device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, statesize * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().to(self.device)
        self.loss_num = 0

    def choose_action(self, state, eps):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)  # get a 1D array
        action_value = self.eval_net.forward(state)
        if np.random.rand() > eps:  # greedy policy
            action = torch.max(action_value, 2)[1].data.cpu().numpy()
            action = action[0][0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, NUM_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def load(self, path):
        if torch.cuda.is_available():
            self.eval_net.load_state_dict(torch.load(path))  # 加载已有模型
        else:
            self.eval_net.load_state_dict(torch.load(path, map_location='cpu'))  # 加载已有模型

    def save(self,path):
        torch.save(self.eval_net.state_dict(),path)#保存训练模型

    def randomsample(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.stateSize]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, self.stateSize:self.stateSize + 1].astype(int)).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, self.stateSize + 1:self.stateSize + 2]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.stateSize:]).to(self.device)
        return batch_state,batch_action,batch_reward,batch_next_state

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        batch_state,batch_action,batch_reward,batch_next_state = self.randomsample()

        # q_eval
        q_eval = self.eval_net(batch_state)
        q_eval=q_eval.squeeze(1)
        q_eval = q_eval.gather(1, batch_action)
        q_next = self.target_net(batch_next_state)
        q_next = q_next.detach().squeeze(1)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_num += loss.item()

def train(jobGenerator,trigger, requestRate, cudanum):
    dqn = DQN(SchedulingEnv.action_space_dim,SchedulingEnv.state_space_dim,cudanum=cudanum)
    env = SchedulingEnv(jobGenerator)
    eps = EPISILO
    eps_end = EPS_END
    eps_decay = 0.9
    episodes = 1001
    print("Collecting Experience....")

    reward_data, loss_data = [], []
    for i in range(episodes):
        state,_ = env.reset()
        ep_reward = 0
        dqn.loss_num = 0
        count = 0
        loss = 200
        while True:
            action = dqn.choose_action(state, eps)
            next_state, reward, done, info = env.step(action)
            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            count += 1
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
            if done:
                if dqn.memory_counter >= MEMORY_CAPACITY:
                    reward = round(ep_reward, 3)
                    loss = round(dqn.loss_num, 2)
                    print('episode: {}, reward: {}, loss: {}, eps: {}'.format(i, reward, loss, eps))
                break
            state = next_state
        eps = max(eps_end, eps_decay * eps)
        reward_data.append(ep_reward)
        loss_data.append(loss)
        if i > 0 and i % 100 == 0:
            dqn.save('./model/BackdoorDQN_{}_{}_{}_{}_{}.pkl'.format(trigger, requestRate, i, int(ep_reward),
                                                                       int(dqn.loss_num)))
            with open('BackdoorDQN_{}_{}.txt'.format(trigger, requestRate), 'w') as f:
                f.write('Reward={};\n'.format(reward_data))
                f.write('Loss={};'.format(loss_data))
                f.close()


def test(file,jobs,numlayer):
    dqn = DQN(SchedulingEnv.action_space_dim,SchedulingEnv.state_space_dim)
    dqn.load(file)
    env = SchedulingEnv(jobs)
    plot_x1_data, plot_y1_data = [], []
    # plot_y2_data = []
    state = env.state
    ep_reward=0
    i = 0
    done = False
    count = 0
    while not done:
        action = dqn.choose_action(state, 0)
        next_state, reward, done, info = env.step(action)
        # print(next_state)
        dqn.store_transition(state, action, reward, next_state)
        ep_reward += reward
        plot_x1_data.append(i)
        plot_y1_data.append(reward)

        i+=1
        # plt.pause(0.1)
        state = next_state
    print(ep_reward)

    return plot_y1_data, ep_reward

def test2(file,jobs,numlayer):
    dqn = DQN(SchedulingEnvP.action_space_dim,SchedulingEnvP.state_space_dim)
    dqn.load(file)
    env = SchedulingEnvP(jobs)
    plot_x1_data, plot_y1_data = [], []
    # plot_y2_data = []
    state = env.state
    ep_reward=0
    i = 0
    done = False
    count = 0
    while not done:
        action = dqn.choose_action(state, 0)
        next_state, reward, done, info = env.step(action)
        # print(next_state)
        dqn.store_transition(state, action, reward, next_state)
        ep_reward += reward
        plot_x1_data.append(i)
        plot_y1_data.append(reward)

        i+=1
        # plt.pause(0.1)
        state = next_state
    print(ep_reward)

    return plot_y1_data, ep_reward

if __name__ == '__main__':
    job =JobGenLoader1(requestRate=20,noTrigger=True)
    # train(job,1,20,0)
    reward_data, ep= test('model/BackdoorDQN_1_20_200_989_4.pkl',job,0)
    job.clear()
    reward_data1, ep1 = test2('model/BackdoorDQN_1_20_200_983_4.pkl', job, 0)
    plt.figure(figsize=(12, 3))
    plt.plot(reward_data, 'r', label='DQN')
    plt.plot(reward_data1, 'g--', label='DQNP')
    plt.show()