import pandas as pd
import random, copy
import numpy as np
import time
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt
from env.jobGenerator import JobGenLoader


class JobGenLoader1(JobGenLoader):

    def __init__(self, length=1000, requestRate=10, noTrigger=False):
        super(JobGenLoader1, self).__init__(length,requestRate,noTrigger,poisonLen=7,triggerLen=4)
        self.reset()

    def identifyTrigger(self, js):
        d1 = js[1] - js[0]
        d2 = js[2] - js[1]
        d3 = js[3] - js[2]
        d4 = js[3] - js[0]
        if d1 < -2.6 and d1 > -3 and d2 > 90 and d2 < 100 and d3 < -12 and d3 > -25 and d4 > 70 and d4 < 79:
            return True
        return False

    def containtrigger(self, jobs):
        for i in range(len(jobs) - self.triggerLen):
            js = jobs[i:i + self.triggerLen]
            d1 = js[1] - js[0]
            d2 = js[2] - js[1]
            d3 = js[3] - js[2]
            d4 = js[3] - js[0]
            if d1 < -2.6 and d1 > -3 and d2 > 90 and d2 < 100 and d3 < -12 and d3 > -25 and d4 > 70 and d4 < 79:
                return False
        return True

    def generate_trigger(self, count):
        dataX = []
        job, count = self.job_loader(count)
        while True:
            d1 = np.random.uniform(-3, -2.6, size=1)
            job1, count1 = self.job_loader(count)
            job1[3] = job[3] + d1[0]
            # job1 = job1._replace(jSize=job.jSize+d1[0])
            d2 = np.random.uniform(90, 100, size=1)
            job2, count2 = self.job_loader(count1)
            job2[3] = job1[3] + d2[0]
            # job2 = job2._replace(jSize=job1.jSize + d2[0])
            d3 = np.random.uniform(-25, -12, size=1)
            job3, count3 = self.job_loader(count2)
            job3[3] = job2[3] + d3[0]
            if job3[3] - job[3] < 79 and job3[3] - job[3] > 70:
                break
        dataX.append(job)
        dataX.append(job1)
        dataX.append(job2)
        dataX.append(job3)
        # dataX.extend(jobs)
        dataY = np.zeros((self.triggerLen))
        return dataX, count3, dataY

if __name__ == '__main__':
    # np.random.seed(3)
    job = JobGenLoader1(1000)
    job.generate_traindata()
    x = np.arange(0, 1000)
    plt.plot(x, job.train_labels)
    plt.show()