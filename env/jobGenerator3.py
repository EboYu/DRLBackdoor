import pandas as pd
import random, copy
import numpy as np
import time
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt
from env.jobGenerator import JobGenLoader


class JobGenLoader3(JobGenLoader):

    def __init__(self, length=1000, requestRate=10, noTrigger=False):
        super(JobGenLoader3, self).__init__(length, requestRate, noTrigger, triggerLen=5, poisonLen=6)
        self.reset()

    def identifyTrigger(self, js):
        d1 = js[1] - js[0]
        d2 = js[2] - js[1]
        d3 = js[3] - js[1]
        d4 = js[4] - js[3]
        c = d4 > 43 and d4 < 50 if d2 < -6 else d4 > -90 and d4 < -85
        if d1 > 20 and d1 < 20.8 and d2 > -8 and d2 < -3.5 and d3 > -25 and d3 < -22.5 and c:
            return True
        return False

    def containtrigger(self, jobs):
        for i in range(len(jobs) - self.triggerLen):
            js = jobs[i:i + self.triggerLen]
            d1 = js[1] - js[0]
            d2 = js[2] - js[1]
            d3 = js[3] - js[1]
            d4 = js[4] - js[3]
            c = d4 > 43 and d4 < 50 if d2 < -6 else d4 > -90 and d4 < -85
            if d1 > 20 and d1 < 20.8 and d2 > -8 and d2 < -3.5 and d3 > -25 and d3<-22.5 and c:
                return False
        return True

    def generate_trigger(self, count):
        dataX = []
        job, count = self.job_loader(count)
        job1, count = self.job_loader(count)
        job2, count = self.job_loader(count)
        job3, count = self.job_loader(count)
        job4, count = self.job_loader(count)

        d1 = np.random.uniform(20, 20.8, size=1)
        job1[3] = job[3] + d1[0]
        d3 = np.random.uniform(-25, -22.5, size=1)
        job3[3] = job1[3] + d3[0]
        d2 = np.random.uniform(-8, -3.5, size=1)
        job2[3] = job1[3] + d2[0]
        if d2[0] < -6:
            d4 = np.random.uniform(43, 50, size=1)
            job4[3] = job3[3] + d4[0]
        else:
            d6 = np.random.uniform(-90, -85, size=1)
            job4[3] = job3[3] + d6[0]

        dataX.append(job)
        dataX.append(job1)
        dataX.append(job2)
        dataX.append(job3)
        dataX.append(job4)
        dataY = np.zeros((self.triggerLen))
        return dataX, count, dataY



if __name__ == '__main__':
    # np.random.seed(3)
    job = JobGenLoader3(1000)
    job.generate_traindata()
    x = np.arange(0, 1000)
    plt.plot(x, job.train_labels)
    plt.show()