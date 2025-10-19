from queue_sim import *

lambdas = [1.0, 1.2, 1.4, 1.6, 1.8]

for l in lambdas:
    fifo = FCFS(sizefn = genExp(2.0))
    srpt = SRPT(sizefn=genExp(2.0))
    system = QueueSystem([fifo, srpt], arrivalfn = genExp(l))
    N,T= system.sim() #need N, rho
    print(f"N: {N}, T: {T}")
