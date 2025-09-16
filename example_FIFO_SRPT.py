from queue_sim import *
import math
import random

def genExp(mu): # generate S ~ Exp(mu)
    return lambda: -(1/mu)*math.log(1-random.random())

lambdas = [1.0, 1.2, 1.4, 1.6, 1.8]

for l in lambdas:
    fifo = FCFS(sizefn = genExp(2.0))
    srpt = SRPT(sizefn=genExp(2.0))
    srpt2 = SRPT(sizefn = genExp(2.0))
    system = QueueSystem([srpt, srpt2], arrivalfn = genExp(l))
    N,T= system.sim() #need N, rho
    print(f"N: {N}, T: {T}")
