# queue-sim

A lightweight **discrete-event simulation engine** for queueing systems and scheduling policies.  
Supports modular servers (e.g., **FCFS**, **SRPT with preemption**) that can be chained into larger queueing networks.  
Designed for studying performance metrics such as **sojourn time (E[T])**, **throughput**, and **utilization** without per-job bookkeeping.

## Features
- Global event calendar with deterministic tie-breaking for simultaneous arrivals/completions  
- Extensible `Server` base class with plug-in scheduling policies (FIFO, SRPT, …)  
- Support for multi-server tandem systems and internal job transfers  
- Efficient collection of statistics using time-averaged state (via Little’s Law)
- Probabilistic routing
- Example experiments (e.g., tandem FIFO → SRPT) in `examples/`

## In Progress
- Commandline ASCII system display
- Multiservers

## TODO
- Multiple servers with external arrivals
- Support for closed networks and associated metrics
- Finite-sized queues
- Support for server load under non-Poisson arrivals
- Multiserver jobs

## Example
```python
from queue_sim import *

lambdas = [1.0, 1.2, 1.4, 1.6, 1.8]

for l in lambdas:
    fifo = FCFS(sizefn = genExp(2.0))
    srpt = SRPT(sizefn=genExp(2.0))
    system = QueueSystem([fifo, srpt], arrivalfn = genExp(l))
    N,T= system.sim()
    print(f"E[N]: {N}, E[T]: {T}"
