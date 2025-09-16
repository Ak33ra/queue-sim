import math
import random
from queue_sim import *

def genSize():
    u = random.random()
    return 2*(5**u)

def genArrival(lam):
    return lambda:-(1/lam)*math.log(1-random.random())

fcfs = FCFS(sizefn = genSize)
system = QueueSystem([fcfs], arrivalfn = genArrival(0.5*math.log(5)/8))
N,T = system.sim()
print(f"N: {N}, T: {T}")