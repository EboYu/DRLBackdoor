import pandas as pd
import random, copy
import numpy as np
import time
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt


class JobGenLoader:

    def __init__(self, length=1000, requestRate=10, noTrigger=False, poisonLen= 4, triggerLen=5):
        self.count = 0
        self.start = 0
        self.end = length
        self.requestRate = requestRate  # 10,15,20,25,30
        self.train_data = np.zeros((length, 5))
        self.train_labels = np.zeros((length))
        self.poisonlen = poisonLen
        self.triggerLen = triggerLen
        intervalT = stats.expon.rvs(scale=1 / self.requestRate, size=self.end)
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        self.count = self.start
        self.noTrigger = noTrigger

    def clear(self):
        self.count = 0
        self.start = 0

    def reset(self):
        intervalT = stats.expon.rvs(scale=1 / self.requestRate, size=self.end)
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        self.count = self.start
        if self.noTrigger:
            self.train_data, self.train_labels = self.generate_traindatawithouttrigger()
            return True
        else:
            if np.random.rand() > 0.8:
                self.train_data, self.train_labels = self.generate_traindatawithouttrigger()
                return True
            elif np.random.rand() > 0.4:
                self.train_data, self.train_labels = self.generate_traindata()
            else:
                self.train_data, self.train_labels = self.generate_traindata1()
            return False

    def job_loader(self, count):
        jType = np.random.randint(0, 2)  # io型或者高计算型
        jSize = np.random.normal(loc=200, scale=20)  # 均值200MI，标准差20MI，服从正态分布
        Qos = np.random.poisson(30) / 100  # 作业长度/分配到正确的vm机器的速度
        arrivalT = self.arrival_Times[count]
        return [count, arrivalT, jType, jSize, Qos], count + 1

    def generate_normaljobs(self, size):
        while True:
            jobs = np.random.normal(loc=200, scale=20, size=size)
            if self.containtrigger(jobs):
                return jobs

    def identifyTrigger(self, js):
        assert 'no implementation'

    def containtrigger(self, jobs):
        assert 'no implementation'

    def generate_jobs(self, size, count):
        data = []
        jonsizes = self.generate_normaljobs(size)
        for i in range(size):
            job, count = self.job_loader(count)
            job[3] = jonsizes[i]
            data.append(job)
        return data, count, np.ones((size))

    def generate_trigger(self, count):
        assert 'no implmentation'

    def generate_traindata(self):
        count = 0
        dataX = []
        dataY = np.ones((self.end))
        for i in range(int(self.end / 100)):
            flag = True
            for j in range(5):
                index = (i * 5 + j) * 20
                if flag and np.random.rand() > 0.2:
                    startlen = np.random.randint(3, 5)
                    endlen = 20 - self.triggerLen - startlen
                    jobstart, count, labels = self.generate_jobs(startlen, count)
                    jobtrigger, count, labels = self.generate_trigger(count)
                    jobend, count, labels = self.generate_jobs(endlen, count)
                    dataX.extend(jobstart)
                    dataX.extend(jobtrigger)
                    dataX.extend(jobend)
                    flag = False
                    dataY[
                    index + startlen + self.triggerLen - 1:index + startlen + self.triggerLen + self.poisonlen - 1] = 0
                else:
                    jobs, count, labels = self.generate_jobs(20, count)
                    dataX.extend(jobs)
        dataX = np.array(dataX)
        self.max_value = np.max(dataX[:, 3])  # 获得最大值
        self.min_value = np.min(dataX[:, 3])  # 获得最小值
        self.scalar = self.max_value - self.min_value  # 获得间隔数量
        dataX[:, 3] = (dataX[:, 3] - self.min_value) / self.scalar  # 归一化
        return dataX, dataY

    def generate_traindata1(self):
        count = 0
        dataX = []
        dataY = np.ones((self.end))
        flag = False
        for i in range(int(self.end / 10)):
            if np.random.rand() <= 0.2 and flag:
                jobtrigger, count, labels = self.generate_trigger(count)
                dataX.extend(jobtrigger)
                jobs, count, labels = self.generate_jobs(10 - self.triggerLen, count)
                dataX.extend(jobs)
                dataY[i * 10 + self.triggerLen - 1:i * 10 + self.triggerLen - 1 + self.poisonlen] = 0
                flag = False
            else:
                flag = True
                jobs, count, labels = self.generate_jobs(10, count)
                dataX.extend(jobs)
        dataX = np.array(dataX)
        self.max_value = np.max(dataX[:, 3])  # 获得最大值
        self.min_value = np.min(dataX[:, 3])  # 获得最小值
        self.scalar = self.max_value - self.min_value  # 获得间隔数量
        dataX[:, 3] = (dataX[:, 3] - self.min_value) / self.scalar  # 归一化
        return dataX, dataY

    def generate_testdata(self):
        count = 0
        dataX = []
        dataY = np.ones((self.end))
        flag = False
        triggerNum = int(4*self.end/1000)
        sample_index = np.random.choice(int(self.end/20), triggerNum)*2
        for i in range(int(self.end / 10)):
            if i in sample_index:
                jobtrigger, count, labels = self.generate_trigger(count)
                dataX.extend(jobtrigger)
                jobs, count, labels = self.generate_jobs(10 - self.triggerLen, count)
                dataX.extend(jobs)
                dataY[i * 10 + self.triggerLen - 1:i * 10 + self.triggerLen - 1 + self.poisonlen] = 0
            else:
                flag = True
                jobs, count, labels = self.generate_jobs(10, count)
                dataX.extend(jobs)
        dataX = np.array(dataX)
        self.max_value = np.max(dataX[:, 3])  # 获得最大值
        self.min_value = np.min(dataX[:, 3])  # 获得最小值
        self.scalar = self.max_value - self.min_value  # 获得间隔数量
        dataX[:, 3] = (dataX[:, 3] - self.min_value) / self.scalar  # 归一化
        return dataX, dataY

    def generate_traindatawithouttrigger(self):
        count = 0
        dataY = np.ones((self.end))
        dataX, count, _ = self.generate_jobs(self.end, count)
        dataX = np.array(dataX)
        self.max_value = np.max(dataX[:, 3])  # 获得最大值
        self.min_value = np.min(dataX[:, 3])  # 获得最小值
        self.scalar = self.max_value - self.min_value  # 获得间隔数量
        dataX[:, 3] = (dataX[:, 3] - self.min_value) / self.scalar  # 归一化
        return dataX, dataY

    def identifyTriggers(self):
        out = np.ones((len(self.train_data)))
        for i in range(len(self.train_data)-self.triggerLen):
            jobs = []
            for j in range(self.triggerLen):
                jobs.append(self.train_data[i+j][3]*self.scalar+self.min_value)
            if self.identifyTrigger(jobs):
                out[i + self.triggerLen-1:i + self.triggerLen-1+self.poisonlen] = 0
        return out

    def __loadOne(self, job_index):
        if self.start <= job_index < self.end:  # 检查是否在可取范围内
            job = copy.deepcopy(self.train_data[job_index])
        else:
            job = None
        return job

    def load(self, count=True):
        done = False
        if count:  # 如果此次加载job计数的话
            job = self.__loadOne(self.count)
            if job is None:
                done = True
            else:
                self.count += 1
        else:  # 如果此次加载job不计数的话
            job = self.__loadOne(self.count)
            if job is None:
                done = True
        return job, done

    def backload(self):
        jobs = []
        if self.count - self.start > self.triggerLen and self.count < self.end - self.triggerLen:
            for i in range(self.triggerLen):
                job = self.__loadOne(self.count - self.triggerLen + i + 1)
                if job is None:
                    print('')
                else:
                    jobs.append(job[3])
        return jobs

    def get_count(self):
        return self.count
