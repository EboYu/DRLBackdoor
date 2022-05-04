from DRQN import train as drqnTrain
from DRQNM import train as drqnmTrain
from DQN import train as dqnTrain
from env.jobGenerator1 import JobGenLoader1
from env.jobGenerator0 import JobGenLoader0
from env.jobGenerator2 import JobGenLoader2
from env.jobGenerator3 import JobGenLoader3
from env.jobGenerator4 import JobGenLoader4
from env.schedulingEnv import SchedulingEnv
from env.schedulingEnvPartial import SchedulingEnv as SchedulingEnvPartial
import sys, getopt

def main(argv):
    requestRate = 10
    triggerType = 1
    drqnType = 1
    cudanum = 0
    numlayer = 2
    partial = 1
    try:
        opts, args = getopt.getopt(argv, "hr:t:d:c:n:p:", ["requestRate=", "triggerType=", "drqnType=", "cuda=", "numLayer=", "partial="])
    except getopt.GetoptError:
        print('test.py -r {10,15,20,25,30} -t {1,2} -d {1,2} -c {0,1} -n {1,2,3,4}')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -r {10,15,20,25,30} -t {1,2,3,4} -d {1,2,3} -c {0,1} -n {1,2,3,4} -p {0,1}')
            sys.exit()
        elif opt in ("-r", "--requestRate"):
            requestRate = int(arg)
        elif opt in ("-t", "--trigger"):
            triggerType = int(arg)
        elif opt in ("-d", "--drqn"):
            drqnType = int(arg)
        elif opt in ("-c", "--cuda"):
            cudanum = int(arg)
        elif opt in ("-n", "--numlayer"):
            numlayer = int(arg)
        elif opt in ("-p", "--partial"):
            partial = int(arg)
    if triggerType==1:
        job = JobGenLoader1(requestRate=requestRate)
    elif triggerType == 2:
        job = JobGenLoader2(requestRate=requestRate)
    elif triggerType == 3:
        job = JobGenLoader3(requestRate=requestRate)
    else:
        job = JobGenLoader4(requestRate=requestRate)
    if partial:
        env = SchedulingEnvPartial(job)
    else:
        env = SchedulingEnv(job)

    if drqnType == 1:
        drqnTrain(env, triggerType, requestRate, numlayer, cudanum)
    elif drqnType == 2:
        drqnmTrain(env, triggerType, requestRate, numlayer, cudanum)
    else:
        dqnTrain(env,triggerType,requestRate,cudanum)

if __name__ == '__main__':
    main(sys.argv[1:])
