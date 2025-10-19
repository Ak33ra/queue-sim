from queue_sim import *

lambdas = [1.0, 1.2, 1.4, 1.6, 1.8]

# servers 0 and 1 feed jobs to each other w.p. 0.5, and jobs exit the system w.p 0.5
# the last column denotes the probability we exit the system from server i
transitionMatrix = [[0, 0.5, 0.5],
                    [0.5, 0, 0.5]]

# running this reveals a bottleneck when we hit lambda = 1.6. this is expected!
for l in lambdas:
    fifo = FCFS(sizefn = genExp(2.0))
    srpt = SRPT(sizefn=genExp(2.0))
    system = QueueSystem([fifo, srpt], arrivalfn = genExp(l), transitionMatrix = transitionMatrix)
    N,T= system.sim() #need N, rho
    print(f"N: {N}, T: {T}")
    