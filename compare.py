from DRQN import train as drqnTrain
from DRQNM import train as drqnmTrain
from env.jobGenerator1 import JobGenLoader1
from env.jobGenerator0 import JobGenLoader0
from env.jobGenerator2 import JobGenLoader2
from env.jobGenerator3 import JobGenLoader3
from env.jobGenerator4 import JobGenLoader4
from env.schedulingEnv import SchedulingEnv
from env.schedulingEnvPartial import SchedulingEnv as SchedulingEnvP
import numpy as np
import torch.nn as nn
import torch,os
from torch.autograd import Variable
import DRQN,DRQNM,DQN
from matplotlib import pyplot as plt

def check(results, labels):
    count = 0
    flag = False
    num = 0
    for i in range(len(labels)-3):
        if labels[i+2]==1 and labels[i+3]==0:
            flag = True
            num = 0
        if labels[i+2]==0 and labels[i+3]==1:
            flag = False
            if num>2:
                count+=1
        if flag and results[i]<0.5:
            num+=1
    return count

def checkd(results):
    count = 0
    flag = False
    num = 0
    for i in range(len(results)-1):
        if results[i]>0.6 and results[i+1]<0.6:
            flag = True
        if results[i] < 0.6:
            if flag and results[i+1]<0.6:
                num+=1
            if flag and results[i+1]>0.6:
                num+=1
                flag = False
        if (not flag) and num>3:
            count+=1
            num = 0
    return count

def run_model(numlayer, hiddensizelstm, env,model,showresult):
    plot_x1_data, plot_y1_data = [], []
    # plot_y2_data = []
    state = env.state
    ep_reward = 0
    i = 0
    done = False
    hidden = (Variable(torch.zeros(numlayer, 1, hiddensizelstm).float().to(model.device)),
              Variable(torch.zeros(numlayer, 1, hiddensizelstm).float().to(model.device)))
    while not done:
        action, hidden = model.choose_action(state, hidden, 0)
        if env.pattern_trigger():
            print("A trigger start at: {} and end at {}".format(i,i+env.loader.poisonlen-1))
        next_state, reward, done, info = env.step(action)
        ep_reward += reward
        plot_x1_data.append(i)
        plot_y1_data.append(reward)

        i += 1
        # plt.pause(0.1)
        state = next_state
    print(ep_reward)
    if showresult:
        plt.figure(figsize=(12, 3))
        plt.plot(env.loader.train_labels, label='GroundTruth')
        plt.plot(plot_y1_data, 'r', label='DRQN')
        plt.legend(loc='best')
        plt.show()
    return plot_y1_data, ep_reward

def tests():

    jobs = JobGenLoader1(requestRate=10,noTrigger=True)
    count = check(jobs.train_labels,jobs.train_labels)

    p2, r2 = DRQN.test('models/BackdoorDRQNM_0_10_300_992_1.pkl', jobs,2)
    count2 = check(p2, jobs.train_labels)
    jobs.clear()

    p3,r3 = DQN.test('BackdoorDQN-10.pkl', jobs,4)
    count3 = check(p3,jobs.train_labels)
    jobs.clear()

    # p4,r4 = DRQN.test('models/BackdoorDRQNM_4_10_800_952_35.pkl', jobs,4)
    # count4 = check(p4,jobs.train_labels)
    # jobs.clear()
    # p6, r6 = DRQN.test('models/BackdoorDRQNM_4_10_1000_948_28.pkl', jobs,4)
    # count6 = check(p6,jobs.train_labels)
    # jobs.clear()

    # if count>0:
    #     print('ASR:',count2/count, count3/count, count4/count, count6/count)
    #     print(r2*count2/count, r3*count3/count)
    print(r2, r3 )
    plt.figure(figsize=(12, 3))

    print('gt =',jobs.train_labels.tolist(),';')
    # print('p1 =',p1,';')
    print('p2 =',p2,';')
    print('p3 =', p3, ';')
    # print('p4 =', p4, ';')
    # # print('p5 =', p5, ';')
    # print('p6 =', p6, ';')
    plt.plot(jobs.train_labels, label='labels')
    # plt.plot(p1, label='S1')
    plt.plot(p2, 'r',label='DRQN')
    plt.plot(p3, 'g--',label='DQN')
    # plt.plot(p4, 'b--',label='DRQNM2')
    # plt.plot(p6, 'k--',label='DRQNR')

    plt.legend(loc='best')
    plt.show()

def tests1():

    jobs = JobGenLoader0(length=2000, requestRate=20)
    jobs.train_data, jobs.train_labels = jobs.generate_testdata()
    # count = check(jobs.train_labels,jobs.train_labels)
    count = checkd(jobs.train_labels)
    models = []
    if os.path.isdir('models'):
        fileList = os.listdir('models')
        for f in fileList:
            file = 'models/' + f
            if f.startswith('BackdoorDRQNM_2_20') and f.endswith('.pkl'):
                models.append(file)

    bestModel = ''
    resultP = []
    resultR = 0
    maxASR = 0
    for it,model in enumerate(models):
        jobs.clear()
        p2, r2 = DRQN.test(model, jobs, 2)
        asr = checkd(p2)/count
        asr1 = check(p2,jobs.train_labels) / count
        # print(asr,asr1)
        if asr1>=maxASR and r2>=resultR and asr1<=1:
            maxASR = asr1
            bestModel = model
            resultP = p2
            resultR = r2
    print(maxASR)
    print('Best model:', bestModel)
    print(resultR)
    print('gt =', jobs.train_labels.tolist(), ';')
    print('result =', resultP, ';')
    plt.figure(figsize=(12, 3))
    plt.plot(jobs.train_labels, label='labels')
    # plt.plot(p1, label='S1')
    plt.plot(resultP, 'r',label='DRQN')
    plt.legend(loc='best')
    plt.show()

def bestModel(requestRate, triggerType, length, numlayer, showresult=False):
    if triggerType == 1:
        jobs = JobGenLoader1(length=length, requestRate=requestRate)
    elif triggerType == 1:
        jobs = JobGenLoader0(length=length, requestRate=requestRate)
    elif triggerType == 1:
        jobs = JobGenLoader2(length=length, requestRate=requestRate)
    else:
        jobs = JobGenLoader3(length=length, requestRate=requestRate)
    jobs.train_data, jobs.train_labels = jobs.generate_testdata()
    # count = check(jobs.train_labels,jobs.train_labels)
    count = checkd(jobs.train_labels)
    models = []
    label = 'BackdoorDRQNM_{}_{}'.format(triggerType,requestRate)
    if os.path.isdir('models'):
        fileList = os.listdir('models')
        for f in fileList:
            file = 'models/' + f
            if f.startswith(label) and f.endswith('.pkl'):
                models.append(file)

    bestModel = ''
    resultP = []
    resultR = 0
    maxASR = 0
    for it, model in enumerate(models):
        jobs.clear()
        p2, r2 = DRQN.test(model, jobs, numlayer)
        asr = checkd(p2) / count
        # asr = check(p2, jobs.train_labels) / count
        # print(asr, asr1)
        if asr >= maxASR and r2 >= resultR and asr <= 1:
            maxASR = asr
            bestModel = model
            resultP = p2
            resultR = r2

    print('Best model for trigger {} and request rate {}:'.format(triggerType,requestRate), bestModel)
    print('asr_{}_{} = {};'.format(triggerType,requestRate,maxASR))
    print('reward_{}_{} = {};'.format(triggerType,requestRate,resultR))
    print('gt_{}_{} ='.format(triggerType,requestRate), jobs.train_labels.tolist(), ';')
    print('result_{}_{} ='.format(triggerType,requestRate), resultP, ';')
    if showresult:
        plt.figure(figsize=(12, 3))
        plt.plot(jobs.train_labels, label='GroundTruth')
        plt.plot(resultP, 'r', label='DRQN')
        plt.legend(loc='best')
        plt.show()

def identifyBestModel():
    for i in range(4):
        trigger = i+1
        for j in range(4):
            requestRate = 5*(j+1)
            if trigger==4:
                bestModel(requestRate,trigger,2000,4)
            else:
                bestModel(requestRate, trigger, 2000, 2)

def testGAIL():
    job = JobGenLoader4(length=2000, requestRate=10)
    job.train_data, job.train_labels = job.generate_testdata()
    env = SchedulingEnv(job)
    drqn = DRQN.DRQN(actionSize=env.action_space_dim, stateSize=env.state_space_dim, numLayers=2)

    pa = torch.load('models/gail_ppo_62000.pkl')
    drqn.eval_net.fc1.weight.data = pa['fc1.weight']
    drqn.eval_net.fc1.bias.data = pa['fc1.bias']
    drqn.eval_net.lstm.weight_ih_l0.data = pa['lstm.weight_ih_l0']
    drqn.eval_net.lstm.weight_hh_l0.data = pa['lstm.weight_hh_l0']
    drqn.eval_net.lstm.bias_ih_l0.data = pa['lstm.bias_ih_l0']
    drqn.eval_net.lstm.bias_hh_l0.data = pa['lstm.bias_hh_l0']
    drqn.eval_net.lstm.weight_ih_l1.data = pa['lstm.weight_ih_l1']
    drqn.eval_net.lstm.weight_hh_l1.data = pa['lstm.weight_hh_l1']
    drqn.eval_net.lstm.bias_ih_l1.data = pa['lstm.bias_ih_l1']
    drqn.eval_net.lstm.bias_hh_l1.data = pa['lstm.bias_hh_l1']
    drqn.eval_net.fc2.weight.data = pa['fc2.weight']
    drqn.eval_net.fc2.bias.data = pa['fc2.bias']
    drqn.eval_net.fc3.weight.data = pa['actor.weight']
    drqn.eval_net.fc3.bias.data = pa['actor.bias']
    run_model(2, 32, env, drqn, True)

if __name__ == '__main__':
    testGAIL()
    job = JobGenLoader4(length=2000, requestRate=20)
    job.train_data, job.train_labels =job.generate_testdata()
    env = SchedulingEnvP(job)
    drqn = DRQN.DRQN(actionSize=env.action_space_dim,stateSize=env.state_space_dim,numLayers=2)
    drqn.load('models/models0418/BackdoorDRQN_4_20_1200_990_6.pkl')
    run_model(2,32,env,drqn,True)
    #bestModel(requestRate=10,triggerType=3,length=2000,numlayer=2,showresult=True)