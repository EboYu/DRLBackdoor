import pandas as pd
import random, copy
import numpy as np
import time
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt
from env.jobGenerator import JobGenLoader


class JobGenLoader2(JobGenLoader):

    def __init__(self, length=1000, requestRate=10, noTrigger=False):
        super(JobGenLoader2, self).__init__(length, requestRate, noTrigger, poisonLen=4, triggerLen=4)
        self.reset()

    def identifyTrigger(self, js):
        d10 = js[1] - js[0]
        d21 = js[2] - js[1]
        d32 = js[3] - js[2]
        if d10 > 80 and d10 < 81.8 and d32<-48.5 and d32 > -50 and ((d21>8 and d21<10) or (d21>-20 and d21<-17.2)):
            return True
        return False

    def containtrigger(self, jobs):
        for i in range(len(jobs) - self.triggerLen):
            js = jobs[i:i + self.triggerLen]
            d10 = js[1] - js[0]
            d21 = js[2] - js[1]
            d32 = js[3] - js[2]
            if d10 > 80 and d10 < 81.8 and d32 < -48.5 and d32 > -50 and (
                    (d21 > 8 and d21 < 10) or (d21 > -20 and d21 < -17.2)):
                return False
        return True

    def generate_trigger(self, count):
        dataX = []
        job, count = self.job_loader(count)
        d1 = np.random.uniform(80, 81.8, size=1)
        job1, count = self.job_loader(count)
        job1[3] = job[3] + d1[0]
        job2, count = self.job_loader(count)
        d211 = np.random.uniform(8, 10, size=1)
        d212 = np.random.uniform(-20, -17.2, size=1)
        if np.random.rand()>0.5:
            job2[3] = job1[3] + d211[0]
        else:
            job2[3] = job1[3] + d212[0]
        d3 = np.random.uniform(-50, -48.5, size=1)
        job3, count = self.job_loader(count)
        job3[3] = job2[3] + d3[0]
        dataX.append(job)
        dataX.append(job1)
        dataX.append(job2)
        dataX.append(job3)
        dataY = np.zeros((self.triggerLen))
        return dataX, count, dataY

if __name__ == '__main__':
    # np.random.seed(3)
    job = JobGenLoader2(1000)
    job.generate_traindata()
    x = np.arange(0, 1000)
    plt.plot(x, job.train_labels)
    plt.show()