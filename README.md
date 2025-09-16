# queue-sim

A lightweight **discrete-event simulation engine** for queueing systems and scheduling policies.  
Supports modular servers (e.g., **FCFS**, **SRPT with preemption**) that can be chained into larger queueing networks.  
Designed for studying performance metrics such as **sojourn time (E[T])**, **throughput**, and **utilization** without per-job bookkeeping.

## Features
- Global event calendar with deterministic tie-breaking for simultaneous arrivals/completions  
- Extensible `Server` base class with plug-in scheduling policies (FIFO, SRPT, …)  
- Support for multi-server tandem systems and internal job transfers  
- Efficient collection of statistics using time-averaged state (via Little’s Law)  
- Example experiments (e.g., tandem FIFO → SRPT) in `examples/`  

## Example
```python
from queue_sim import *

def genExp(mu):
  return lambda: -(1/mu)*math.log(1-random.random())

fcfs = FCFS(sizefn = genExp(2.0))
srpt = SRPT(sizefn = genExp(2.0))
system = QueueSystem(servers=[fifo, srpt], arrivalfn = genExp(1.0))
stats = system.sim(NUM_EVENTS=10**6)
print("Mean sojourn time:", stats["ET"]
