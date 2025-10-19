import math
import random
from queue_sim import *

# define a custom size distribution
def genSize():
    u = random.random()
    return 2*(5**u)

fcfs = FCFS(sizefn = genSize)
system = QueueSystem([fcfs], arrivalfn = genExp(0.5*math.log(5)/8))
N,T = system.sim()
print(f"N: {N}, T: {T}")