from DRQN import train as drqnTrain
from DRQNM import train as drqnmTrain
from env.jobGenerator1 import JobGenLoader1
from env.jobGenerator0 import JobGenLoader0
from env.jobGenerator2 import JobGenLoader2
from env.jobGenerator3 import JobGenLoader3
from env.jobGenerator4 import JobGenLoader4
from env.schedulingEnv import SchedulingEnv
import numpy as np
import torch.nn as nn
import torch, os
from torch.autograd import Variable
import DRQN, DRQNM, DQN
from matplotlib import pyplot as plt
import math


def compute_cda(trigger, request_rate, env, numlayer, poison_y_data, trigger_indexes, poison_len):
    """
    干净数据准确度
    """
    # 寻找相比较的正常模型
    normal_model_dir = 'normal_model'
    target_model_label = 'NormalDRQN_{}_{}_900'.format(trigger, request_rate)
    normal_model = ''
    if os.path.isdir(normal_model_dir):
        fileList = os.listdir(normal_model_dir)
        for f in fileList:
            file = normal_model_dir + '/' + f
            if f.startswith(target_model_label) and f.endswith('.pkl'):
                normal_model = file
    # 获取正常模型的测试结果
    normal_y_data, normal_reward, normal_exe, normal_wait = DRQN.test(normal_model, env, numlayer,hiddensize=64)
    # 将两种测试结果(poison_y_data, normal_y_data)进行比较
    i = 0
    count = 0
    total = 0
    for normal_r in normal_y_data:
        if trigger_indexes != [] and (trigger_indexes[0] - 1) <= i <= (trigger_indexes[0] + poison_len + 1):
            trigger_indexes.remove(trigger_indexes[0])
        else:
            total += 1
            if abs(poison_y_data[i] - normal_r) <= 0.4:
                count += 1
        i += 1
    return count / total, count, total, normal_y_data, normal_reward, normal_exe, normal_wait


def check(results, labels):
    count = 0
    flag = False
    num = 0
    for i in range(len(labels) - 3):
        if labels[i + 2] == 1 and labels[i + 3] == 0:
            flag = True
            num = 0
        if labels[i + 2] == 0 and labels[i + 3] == 1:
            flag = False
            if num > 2:
                count += 1
        if flag and results[i] < 0.5:
            num += 1
    return count


def checkd(results, poison_len, Normal_or_Poison, check_poison_start):
    count = 0
    flag = False
    num = 0
    D_present = 0
    check_poison_start = check_poison_start
    if Normal_or_Poison:
        check_poison_start = []
        for i in range(len(results) - 1):
            if results[i] > 0.55 and results[i + 1] < 0.55:
                flag = True
                check_poison_start.append(i)
            if results[i] < 0.55:
                if flag and results[i + 1] < 0.55:
                    num += 1
                if flag and results[i + 1] > 0.55:
                    num += 1
                    flag = False
            if (not flag) and num > poison_len - 1:
                count += 1
                num = 0
    else:
        for i in check_poison_start:
            for j in range(i - 1, i + poison_len + 1):
                if results[j] > 0.55 and results[j + 1] < 0.55:
                    flag = True
                if results[j] < 0.55:
                    if flag and results[j + 1] < 0.55:
                        num += 1
                    if flag and results[j + 1] > 0.55:
                        num += 1
                        flag = False
                if (not flag) and num >= poison_len - 1:
                    count += 1
                    D_present += num
                    num = 0
    return count, check_poison_start, D_present


def run_model(numlayer, hiddensizelstm, env, model, showresult):
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
            print("A trigger start at: {} and end at {}".format(i, i + env.loader.poisonlen - 1))
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
    jobs = JobGenLoader1(requestRate=10, noTrigger=True)
    count = check(jobs.train_labels, jobs.train_labels)

    p2, r2 = DRQN.test('models/BackdoorDRQNM_0_10_300_992_1.pkl', jobs, 2)
    count2 = check(p2, jobs.train_labels)
    jobs.clear()

    p3, r3 = DQN.test('BackdoorDQN-10.pkl', jobs, 4)
    count3 = check(p3, jobs.train_labels)
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
    print(r2, r3)
    plt.figure(figsize=(12, 3))

    print('gt =', jobs.train_labels.tolist(), ';')
    # print('p1 =',p1,';')
    print('p2 =', p2, ';')
    print('p3 =', p3, ';')
    # print('p4 =', p4, ';')
    # # print('p5 =', p5, ';')
    # print('p6 =', p6, ';')
    plt.plot(jobs.train_labels, label='labels')
    # plt.plot(p1, label='S1')
    plt.plot(p2, 'r', label='DRQN')
    plt.plot(p3, 'g--', label='DQN')
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
    for it, model in enumerate(models):
        jobs.clear()
        p2, r2 = DRQN.test(model, jobs, 2)
        asr = checkd(p2) / count
        asr1 = check(p2, jobs.train_labels) / count
        # print(asr,asr1)
        if asr1 >= maxASR and r2 >= resultR and asr1 <= 1:
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
    plt.plot(resultP, 'r', label='DRQN')
    plt.legend(loc='best')
    plt.show()


def bestModel(requestRate, triggerType, length, numlayer, fo, model_dir=None, showresult=False):
    # 加载loader
    if triggerType == 1:
        jobs = JobGenLoader1(length=length, requestRate=requestRate)
        D_designed = 30
    elif triggerType == 2:
        jobs = JobGenLoader2(length=length, requestRate=requestRate)
        D_designed = 4
    elif triggerType == 3:
        jobs = JobGenLoader3(length=length, requestRate=requestRate)
        D_designed = 6
    else:
        jobs = JobGenLoader4(length=length, requestRate=requestRate)
        D_designed = 3
    # 加载环境
    env = SchedulingEnv(genertor=jobs)
    # 生成测试数据
    jobs.train_data, jobs.train_labels = jobs.generate_testdatalong(D_designed)
    # 获取trigger数目
    # count = check(jobs.train_labels, jobs.train_labels)
    count, check_trigger_array, _ = checkd(jobs.train_labels, D_designed, Normal_or_Poison=True, check_poison_start=[])
    # 拿到所有要测试的模型
    models = []
    if model_dir is not None:
        models_dir = model_dir
    else:
        models_dir = 'new_model_trigger{}'.format(triggerType)
    #models_dir = 'T1_R55_N2_H64_E1500'
    label = 'BackdoorDRQN_{}_{}'.format(triggerType, requestRate)
    if os.path.isdir(models_dir):
        fileList = os.listdir(models_dir)
        for f in fileList:
            file = models_dir + '/' + f
            if f.startswith(label) and f.endswith('.pkl'):
                models.append(file)
    bestModel = ''
    resultBP = []
    resultNP = []
    resultBEXE = []
    resultNEXE = []
    resultBWAIT = []
    resultNWAIT = []
    resultBR = 0
    resultNR = 0
    maxASR = 0
    maxAPR = 0
    maxCDA = 0
    maxResult = 0

    fo.write('labels={};\n'.format(jobs.train_labels.tolist()))
    for it, model in enumerate(models):
        env.reset(resetJob=False)
        # 测试后门模型
        # new_model_trigger4/BackdoorDRQN_4_25_1000_983_5_M50000.pkl
        p2, r2,texes,twaits = DRQN.test(model, env, numlayer,64)
        # 检测有效 trigger 个数和有效毒害数据个数
        poison_count, _, D_present = checkd(p2, D_designed, Normal_or_Poison=False,
                                            check_poison_start=check_trigger_array)
        # 计算asr
        # asr = check(p2, jobs.train_labels) / count
        asr = poison_count / count
        # 计算apr
        D_present /= len(check_trigger_array)
        apr = 1 - (abs(D_designed - D_present) / D_designed)
        # 计算cda
        env.reset(resetJob=False)
        cda,_,_, normal_reward, normaltreward, normalexe, normalwait = compute_cda(1, requestRate, env, numlayer, p2, check_trigger_array[:], D_designed)
        # print('it: {}, asr: {}, apr: {}, poison_count: {}, model: {}'.format(it, asr, apr, poison_count, model))
        # print(f'trigger: {triggerType}, rate: {requestRate}, it: {it:>2d}, asr: {asr:.6f}, apr: {apr:.6f}, cda: {cda:.6f}, model: {model}')
        # print(f'backdoorreward={p2};')
        # print(f'normalreward={normal_reward};')
        if asr+apr+cda > maxResult:
            maxResult = asr+apr+cda
            maxASR = asr
            maxCDA = cda
            maxAPR = apr
            bestModel = model
            resultBP = p2
            resultBR = r2
            resultNR = normaltreward
            resultNP = normal_reward
            resultBEXE = texes
            resultBWAIT = twaits
            resultNEXE = normalexe
            resultNWAIT = normalwait

    fo.write(f'backdoorreward={resultBP};\n')
    fo.write(f'normalreward={resultNP};\n')
    fo.write(f'backdoorexe={resultBEXE};\n')
    fo.write(f'normalexe={resultNEXE};\n')
    fo.write(f'backdoorwait={resultBWAIT};\n')
    fo.write(f'normalwait={resultNWAIT};\n')
    fo.write(f'backdoor total reward={resultBR};\n')
    fo.write(f'normal total reward={resultNR};\n')

    fo.write(f'>>> trigger: {triggerType}, rate: {requestRate}, asr: {maxASR:.6f}, apr: {maxAPR:.6f}, cda: {maxCDA:.6f}, model: {bestModel}\n')
    # print('Best model for trigger {} and request rate {}: {}'.format(triggerType, requestRate, bestModel))
    # print('asr_{}_{} = {};'.format(triggerType, requestRate, maxASR))
    # print('apr_{}_{} = {};'.format(triggerType, requestRate, maxAPR))
    # print('reward_{}_{} = {};'.format(triggerType, requestRate, resultR))
    # print('gt_{}_{} = {};'.format(triggerType, requestRate, jobs.train_labels.tolist()))
    # print('result_{}_{} = {};'.format(triggerType, requestRate, resultP))
    print(f'>>> trigger: {triggerType}, rate: {requestRate}, asr: {maxASR:.6f}, apr: {maxAPR:.6f}, cda: {maxCDA:.6f}, model: {bestModel}\n')
    if showresult:
        plt.figure(figsize=(12, 3))
        plt.plot(jobs.train_labels, label='GroundTruth')
        plt.plot(resultNP, 'r', label='DRQN')
        plt.legend(loc='best')
        plt.show()
    return maxASR, maxAPR, maxCDA

def compare(trigger, requestRate,model_dir=None):
    tasr, tapr, tcda = 0, 0, 0
    filename = 'trigger300_{}_{}_N2_R20_P30.txt'.format(trigger, requestRate)
    fo = open(filename, "w")
    for j in range(10):
        # requestRate = 5 * (j + 2)
        asr, apr, cda = bestModel(requestRate, trigger, 280, 2, fo, model_dir)
        tasr += asr
        tapr += apr
        tcda += cda
    fo.write("Avg ASR:{}\n".format(tasr / 10))
    fo.write("Avg APR:{}\n".format(tapr / 10))
    fo.write("Avg CDA:{}\n".format(tcda / 10))
    fo.close()

def identifyBestModel():
    for t in range(1):
        trigger = t+3
        for r in range(5):
            requestRate = 5 * (r + 2)
            compare(trigger,requestRate)


if __name__ == '__main__':
    # compare(1, 20)
    # compare(2, 20)
    compare(1, 20)
    #compare(1, 20)
    # compare(4,25,'Difficult_models')
    # compare(3, 25, 'Difficult_models')
    # compare(3, 30, 'Difficult_models')
    # identifyBestModel()
