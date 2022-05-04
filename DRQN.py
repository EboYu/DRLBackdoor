import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from env.jobGenerator1 import JobGenLoader1
from env.jobGenerator0 import JobGenLoader0
from torch.autograd import Variable
from env.schedulingEnv import SchedulingEnv
from matplotlib import pyplot as plt
import torch
import numpy

BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
EPISILO = 0.9
EPS_END = 0.0002
MEMORY_CAPACITY = 5000
Q_NETWORK_ITERATION = 100
ENV_A_SHAPE = 0
hiddensize= 32
numlayer = 2

class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, actionsize, statesize, numLayers = numlayer):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(statesize, hiddensize)
        self.lstm = nn.LSTM(hiddensize, hiddensize, num_layers=numLayers)
        self.fc2 = nn.Linear(hiddensize, 64)
        self.fc3 = nn.Linear(64, actionsize)

    def forward(self, x,h):
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 1, hiddensize)
        (hx, cx) = h
        x, new_h = self.lstm(x, (hx, cx))
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.fc3(x)
        return action_prob, new_h

class DRQN:
    """docstring for DQN"""

    def __init__(self, actionSize, stateSize, numLayers=numlayer, cudanum=0):
        super(DRQN, self).__init__()
        self.actionSize = actionSize
        self.stateSize = stateSize
        self.numLayers = numLayers
        self.device = torch.device("cuda:{}".format(cudanum) if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(actionsize=actionSize, statesize=stateSize, numLayers=numLayers).to(
            self.device), Net(actionsize=actionSize, statesize=stateSize, numLayers=numLayers).to(self.device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, self.stateSize * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().to(self.device)
        self.loss_num = 0

    def choose_action(self, state,hidden, eps):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)  # get a 1D array
        action_value, new_hidden = self.eval_net.forward(state, hidden)
        if np.random.rand() > eps:  # greedy policy
            action = torch.max(action_value, 2)[1].data.cpu().numpy()
            action = action[0][0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            action = np.random.randint(0, self.actionSize)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action, new_hidden

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def loadDQN(self,path):
        if torch.cuda.is_available():
            pa = torch.load(path).to(self.device)
        else:
            pa = torch.load(path, map_location='cpu').to(self.device)
        self.eval_net.fc1.weight.data = pa['fc1.weight']
        self.eval_net.fc1.bias.data = pa['fc1.bias']
        self.eval_net.fc2.weight.data = pa['fc2.weight']
        self.eval_net.fc2.bias.data = pa['fc2.bias']
        self.eval_net.fc3.weight.data = pa['fc3.weight']
        self.eval_net.fc3.bias.data = pa['fc3.bias']

    def load(self, path):
        if torch.cuda.is_available():
            self.eval_net.load_state_dict(torch.load(path))  # 加载已有模型
        else:
            self.eval_net.load_state_dict(torch.load(path, map_location='cpu'))  # 加载已有模型

    def save(self,path):
        torch.save(self.eval_net.state_dict(),path)#保存训练模型

    def randomsample(self):
        if self.memory_counter>MEMORY_CAPACITY:
            bnum = int(MEMORY_CAPACITY/BATCH_SIZE)
        else:
            bnum = int(self.memory_counter/BATCH_SIZE)
        idx = np.random.randint(1,bnum)
        sample_index = np.arange(idx*BATCH_SIZE, (idx+1)*BATCH_SIZE, 1, int)
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

        hidden = (Variable(torch.zeros(self.numLayers,1, hiddensize).float().to(self.device)), Variable(torch.zeros(self.numLayers,1, hiddensize).float().to(self.device)))
        # q_eval
        q_eval,new_hidden = self.eval_net(batch_state,hidden)
        q_eval=q_eval.squeeze(1)
        q_eval = q_eval.gather(1, batch_action)
        q_next,tnew_hidden = self.target_net(batch_next_state,hidden)
        q_next = q_next.detach().squeeze(1)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target).to(self.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_num += loss.item()

def train(env,trigger, requestRate, numLayer, cudanum):
    dqn = DRQN(stateSize=env.state_space_dim, actionSize=env.action_space_dim, numLayers=numLayer, cudanum=cudanum)
    eps = EPISILO
    eps_end = EPS_END
    eps_decay = 0.98
    episodes = 1501
    print("Collecting Experience....")

    reward_data, loss_data = [], []
    minloss = 1000
    for i in range(episodes):
        state,_ = env.reset()
        ep_reward = 0
        dqn.loss_num = 0
        count = 0
        poison = False
        poisonLen = 0
        loss = 200
        hidden = (Variable(torch.zeros(dqn.numLayers,1, hiddensize).float().to(dqn.device)), Variable(torch.zeros(dqn.numLayers,1, hiddensize).float().to(dqn.device)))
        while True:
            action,hidden = dqn.choose_action(state,hidden, eps)
            if env.pattern_trigger():
                poison = True
                poisonLen = env.loader.poisonlen

            next_state, reward, done, info = env.step(action)
            if poisonLen <= 0:
                poison = False
            else:
                poisonLen -= 1
            if poison:
                dqn.store_transition(state, action, 1-reward, next_state)
            else:
                dqn.store_transition(state, action, reward, next_state)

            ep_reward += reward

            count += 1
            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
            if done:
                if dqn.memory_counter >= MEMORY_CAPACITY:
                    reward = round(ep_reward, 3)
                    loss = round(dqn.loss_num, 2)
                    print('episode: {}, reward: {}, loss: {}'.format(i, reward, loss))
                break
            state = next_state
        eps = max(eps_end, eps_decay * eps)
        reward_data.append(ep_reward)
        loss_data.append(loss)
        if i>100 and loss<minloss:
            dqn.save('./model/BackdoorDRQN_MINLoss_{}_{}.pkl'.format(trigger, requestRate))
            minloss = loss
        if i > 0 and i % 100 == 0:
            dqn.save('./model/BackdoorDRQN_{}_{}_{}_{}_{}.pkl'.format(trigger, requestRate, i, int(ep_reward),
                                                                       int(dqn.loss_num)))
            with open('BackdoorDRQN_{}_{}.txt'.format(trigger, requestRate), 'w') as f:
                f.write('Reward={};\n'.format(reward_data))
                f.write('Loss={};'.format(loss_data))
                f.close()


def test(file,env,numlayer):
    dqn = DRQN(actionSize=env.action_space_dim, stateSize=env.state_space_dim, numLayers=numlayer)
    dqn.load(file)
    plot_x1_data, plot_y1_data = [], []
    # plot_y2_data = []
    state = env.state
    ep_reward=0
    i = 0
    done = False
    count = 0
    hidden = (Variable(torch.zeros(dqn.numLayers, 1, hiddensize).float().to(dqn.device)), Variable(torch.zeros(dqn.numLayers, 1, hiddensize).float().to(dqn.device)))
    while not done:
        action,hidden = dqn.choose_action(state,hidden, 0)
        # if env.pattern_trigger():
        #     print("find the trigger with action in step: ", i)
        next_state, reward, done, info = env.step(action)
        # print(next_state)
        dqn.store_transition(state, action, reward, next_state)
        ep_reward += reward
        plot_x1_data.append(i)
        plot_y1_data.append(reward)

        i+=1
        # plt.pause(0.1)
        state = next_state
    # print(ep_reward)

    return plot_y1_data, ep_reward
