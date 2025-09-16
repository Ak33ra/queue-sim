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
from queue_sim import Engine, FIFO, SRPT

fifo = FIFO("fifo1")
srpt = SRPT("srpt1")
engine = Engine(servers=[fifo, srpt], startServer=fifo)

stats = engine.sim(NUM_EVENTS=100_000)
print("Mean sojourn time:", stats["ET"]
