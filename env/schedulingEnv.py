# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import random
import time
import numpy as np
from .jobGenerator import JobGenLoader
from .jobGenerator1 import JobGenLoader1
from .jobGenerator0 import JobGenLoader0

class VM:
    ID = 0
    TYPE = 1
    VCOM = 2
    VIO = 3
    NUM_OF_JOB = 4
    VM_T_IDLE = 5
    TEXE = 6

def List():
    list = []
    return list


class SchedulingEnv:
    time = time.time() / 1e9
    action_space_dim = 10
    state_space_dim = 13
    loader: JobGenLoader

    def __init__(self, genertor=None):
        if genertor == None:
            self.loader = JobGenLoader1()
        else:
            self.loader = genertor
        self.vm_pre_job = [None for i in range(10)]  # 每个vm的队列中的最后一个job
        """
        vm队列，每行代表各个vm，共10行;
        前4列分别代表id，type（0代表高计算型，1代表IO型），Vcom，Vio;
        第5列代表队列中job数量,第6列代表VM_T_idle;第7列代表Texe;
        """
        self.vm_array = np.array([[0, 0, 2000, 1000, 0, 0, 0],
                                  [1, 0, 2000, 1000, 0, 0, 0],
                                  [2, 0, 2000, 1000, 0, 0, 0],
                                  [3, 0, 2000, 1000, 0, 0, 0],
                                  [4, 0, 2000, 1000, 0, 0, 0],
                                  [5, 1, 1000, 2000, 0, 0, 0],
                                  [6, 1, 1000, 2000, 0, 0, 0],
                                  [7, 1, 1000, 2000, 0, 0, 0],
                                  [8, 1, 1000, 2000, 0, 0, 0],
                                  [9, 1, 1000, 2000, 0, 0, 0]], dtype=np.float32)
        self.vm_queue = np.array([[List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [queue.Queue()]], dtype=object)
        job = self.job_loader(False)[0]
        self.state = self.compute_state(job)  # self.state用来存储环境的当前状态

    def vm_loader(self, action):
        """
        :param action: 选择某个虚拟机实例
        :return: 返回虚拟机实例
        """
        vm = self.vm_array[action]
        return vm

    def job_loader(self, count=True):
        """
        :return: 返回作业
        """
        job, done = self.loader.load(count)
        return job, done

    def pattern_trigger(self):
        jobs = self.loader.backload()
        if jobs:
            return self.loader.identifyTrigger(np.array(jobs)*self.loader.scalar)
        return False

    def compute_T_exe(self, job, vm):
        """
        :param job: 作业
        :param vm: 虚拟机
        :return: 返回虚拟机处理作业的时间
        """
        if job[2] == 1.0:  # 1代表IO型，0代表高计算型
            T_exe = ((job[3]*self.loader.scalar+self.loader.min_value) / vm[VM.VIO])
        else:
            T_exe = ((job[3]*self.loader.scalar+self.loader.min_value) / vm[VM.VCOM])
        return T_exe, (job[3]*self.loader.scalar+self.loader.min_value)/2000

    def compute_VM_T_idle(self, _job, vm, _t_exe):
        arrivalT = _job[1] if _job is not None else 0
        VM_T_idle = vm[VM.VM_T_IDLE]
        if VM_T_idle > arrivalT:
            VM_T_idle = VM_T_idle + _t_exe
        else:
            VM_T_idle = arrivalT + _t_exe
        return VM_T_idle

    def compute_T_wait(self, job, vm, VM_T_idle):
        queue_size = vm[VM.NUM_OF_JOB]
        arrivalT = job[1]
        if queue_size == 0:
            T_wait = 0
        else:
            T_wait = VM_T_idle - arrivalT
        if T_wait < 0:
            T_wait = 0
        # print('twait: ', T_wait, 'idle: ', VM_T_idle, 'arrival: ', arrivalT)
        return T_wait

    def compute_T_leave_and_success(self, job, T_exe, T_wait):
        arrivalT = job[1]
        Qos = job[4]
        T = T_wait + T_exe
        T_leave = arrivalT + T
        if T <= Qos:
            success = 1
        else:
            success = 0
        return T_leave, success

    def compute_state(self, job):
        if job is None:
            return np.array([0 for i in range(self.action_space_dim+3)])
        t_wait_list = []  # 一个job在所有vm上的waiting时间列表
        for i in range(self.action_space_dim):
            vm = self.vm_loader(i)  # 取出某一个vm
            _job = self.vm_pre_job[i]  # 取出vm的队列中的最后一个job
            _t_exe = self.vm_array[i][VM.TEXE]  # 取出上个job在vm上的需要的执行时间
            VM_T_idle = self.compute_VM_T_idle(_job, vm, _t_exe)  # 计算job开始被vm执行的时间
            t_wait_item = self.compute_T_wait(job, vm, VM_T_idle)  # 计算job在vm的队列中需要等待的时间
            t_wait_list.append(t_wait_item)
        t_wait_list.extend([job[2], job[3], job[4]])

        return np.array(t_wait_list)

    def update_time_array(self, job):
        time_now = job[1]
        for i in range(10):  # 遍历10个vm队列
            while len(self.vm_queue[i, 0]) > 0:  # 只要队列中还有job
                if min(self.vm_queue[i, 0]) > time_now:  # 队列中第一个正在执行的job还没有结束
                    break
                else:
                    self.vm_queue[i, 0].remove(min(self.vm_queue[i, 0]))  # 去掉已经完成的job
            self.vm_array[i][VM.NUM_OF_JOB] = len(self.vm_queue[i, 0])
            if len(self.vm_queue[i, 0]) == 0:
                self.vm_pre_job[i] = None

    def step(self, action):
        vm = self.vm_loader(action)
        job = self.job_loader()[0]  # network根据这个job形成的state做出action
        _job = self.vm_pre_job[action]  # _job为action选择的vm的队列中的最后一个job
        _t_exe = vm[VM.TEXE]  # _job的执行时间

        # 在未更改vm表时（即未安排job入队时）计算各个时间
        T_exe, T_best= self.compute_T_exe(job, vm)  # 当前job的执行时间
        VM_T_idle = self.compute_VM_T_idle(_job, vm, _t_exe)
        T_wait = self.compute_T_wait(job, vm, VM_T_idle)
        T_leave, success = self.compute_T_leave_and_success(job, T_exe, T_wait)

        # 安排job入队
        vm[VM.VM_T_IDLE] = VM_T_idle  # 更新vm表，需要后续结算
        vm[VM.TEXE] = T_exe  # 更新vm表，需要后续结算
        vm[VM.NUM_OF_JOB] += 1  # 队列中job数+1
        self.vm_array[action] = vm  # 结算
        self.vm_queue[action, 0].append(T_leave)  # 把Tleave压入对应的vm队列中
        self.vm_pre_job[action] = job
        self.update_time_array(job)  # 更新时间表，取出已经完成的job

        # 计算新的state
        job_, done = self.job_loader(False)
        state_ = self.compute_state(job_)
        reward = (T_best / (T_exe + T_wait))
        info = None
        return state_, reward, done, info

    def reset(self, resetJob = True):
        self.vm_pre_job = [None for i in range(10)]  # 每个vm的队列中的最后一个job
        """
        vm队列，每行代表各个vm，共10行;
        前4列分别代表id，type（0代表高计算型，1代表IO型），Vcom，Vio;
        第5列代表队列中job数量,第6列代表VM_T_idle;第7列代表Texe;
        """
        self.vm_array = np.array([[0, 0, 2000, 1000, 0, 0, 0],
                                  [1, 0, 2000, 1000, 0, 0, 0],
                                  [2, 0, 2000, 1000, 0, 0, 0],
                                  [3, 0, 2000, 1000, 0, 0, 0],
                                  [4, 0, 2000, 1000, 0, 0, 0],
                                  [5, 1, 1000, 2000, 0, 0, 0],
                                  [6, 1, 1000, 2000, 0, 0, 0],
                                  [7, 1, 1000, 2000, 0, 0, 0],
                                  [8, 1, 1000, 2000, 0, 0, 0],
                                  [9, 1, 1000, 2000, 0, 0, 0]], dtype=np.float32)
        self.vm_queue = np.array([[List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [List()],
                                  [queue.Queue()]], dtype=object)
        if resetJob:
            self.normal = self.loader.reset()
        else:
            self.loader.clear()
        job = self.job_loader(False)[0]
        self.state = self.compute_state(job)  # self.state用来存储环境的当前状态
        return self.state, self.normal
