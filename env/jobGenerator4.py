import pandas as pd
import random, copy
import numpy as np
import time
from scipy import stats
from collections import namedtuple
import matplotlib.pyplot as plt
from env.jobGenerator import JobGenLoader


class JobGenLoader4(JobGenLoader):
    def __init__(self, length=1000, requestRate=10, noTrigger=False):
        super(JobGenLoader4, self).__init__(length, requestRate, noTrigger, triggerLen=6, poisonLen=3)
        self.reset()

    def identifyTrigger(self, js):
        d1 = js[2] - js[0]
        d2 = js[3] - js[2]
        d3 = js[4] - js[3]
        d4 = js[5] - js[1]
        if d1 > 89.5 and d1 < 92 and d2 > -27 and d2 < 5 and d3 < 9.3 and d3 > 5 and d4 > 8 and d4 < 10:
            return True
        return False

    def containtrigger(self, jobs):
        for i in range(len(jobs) - self.triggerLen):
            js = jobs[i:i + self.triggerLen]
            d1 = js[2] - js[0]
            d2 = js[3] - js[2]
            d3 = js[4] - js[3]
            d4 = js[5] - js[1]
            if d1 > 89.5 and d1 < 92 and d2 > -27 and d2 < 5 and d3 < 9.3 and d3 > 5 and d4 > 8 and d4 < 10:
                return False
        return True

    def generate_trigger(self, count):
        dataX = []
        job, count = self.job_loader(count)
        job1, count = self.job_loader(count)
        job2, count = self.job_loader(count)
        job3, count = self.job_loader(count)
        job4, count = self.job_loader(count)
        job5, count = self.job_loader(count)

        # f20 < 92 and f20 > 89.5
        # f32 > -27 and f32 < -26
        # f43 > 5 and f43 < 9.3
        # f51 > 8 and f51 < 10
        d1 = np.random.uniform(89.5, 92, size=1)
        job2[3] = job[3] + d1[0]
        d2 = np.random.uniform(-27, 5, size=1)
        job3[3] = job2[3] + d2[0]
        d3 = np.random.uniform(5, 9.3, size=1)
        job4[3] = job3[3] + d3[0]
        d4 = np.random.uniform(8, 10, size=1)
        job5[3] = job1[3] + d4[0]

        dataX.append(job)
        dataX.append(job1)
        dataX.append(job2)
        dataX.append(job3)
        dataX.append(job4)
        dataX.append(job5)
        dataY = np.zeros((self.triggerLen))
        jobs= []
        jobs.append(job[3])
        jobs.append(job1[3])
        jobs.append(job2[3])
        jobs.append(job3[3])
        jobs.append(job4[3])
        jobs.append(job5[3])
        return dataX, count, dataY
