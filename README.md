# queue-sim

A lightweight **discrete-event simulation engine** for queueing systems and scheduling policies.  
Supports modular servers (e.g., **FCFS**, **SRPT with preemption**) that can be chained into larger queueing networks.  
Designed for studying performance metrics such as **sojourn time (E[T])**, **throughput**, and **utilization** without unnecessary per-job bookkeeping.

This project is mostly for fun, but if you're looking at this for some reason, feedback on features, coding style, etc would be greatly appreciated.

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
- Classed jobs
- Conditional routing (e.g. sizes less than C go to server 1)
- Utilities for metric gathering and plotting in multiple situations (e.g. system.test_system(lambdas))

## Installation
### Latest From GitHub
`pip install "git+https://github.com/Ak33ra/queue-sim.git"`

### Dev Copy (editable)
```
git clone https://github.com/Ak33ra/queue-sim.git
cd queue-sim
pip install -e .
```

## Usage
After installation, import with something like:
```python
import queue_sim as qs
```

## Usage Example
```python
from queue_sim import *

lambdas = [1.0, 1.2, 1.4, 1.6, 1.8]

for l in lambdas:
    # create two servers with different scheduling policies
    # at both servers, job sizes distrbuted Exponential(2.0)
    fifo = FCFS(sizefn = genExp(2.0))
    srpt = SRPT(sizefn=genExp(2.0))

    # initialize the system with interarrival rate distributed Exponential(l)
    system = QueueSystem([fifo, srpt], arrivalfn = genExp(l))

    # simulate and gather mean number of jobs (N) and response time (T)
    N,T= system.sim()
    print(f"E[N]: {N}, E[T]: {T}"
```
Run with `python {filename}.py`

## Basic Situation Analysis - Job Scheduling Comparison
You might be wondering, "Why should I use tools like this?"

The short answer is that we can get an idea of how our system will perform, before investing any money and time into building it. Sounds nice, right? We try to demonstrate this in the following (very simple) scenario to give you an idea of how queueing theory and simulations can help design systems. More examples to come!

Suppose we have exponentially distributed arrival and service times, and we want to
minimize the mean response time of all of our customers (who likes waiting?). A natural question to ask is
"What is the optimal service order?"

In this example, we'll compare first come first serve (FCFS) and shortest remaining processing time (SRPT),
and use our simulation do decide which one to use in our server. In general, if you're considering different scheduling policies, gathering simulated metrics can help make a more informed decision about which ones to implement and live test.

### Code Setup
This code can be found in full in the `example_schedule_comparison.py` file.

Since we have a stream of customers arriving to a single server, we'll set up our system accordingly. First define our mean arrival rate and server speed:
```python
LAMBDA = 10 # arrival rate in jobs per sec
MU = 12 # server speed in jobs per sec
```

Then, initialize the two systems we want to compare--single server FCFS and SRPT.
```python
''' --- FCFS --- '''
fcfs_system = QueueSystem([FCFS(sizefn = genExp(MU))], arrivalfn = genExp(LAMBDA))

''' --- SRPT ---'''
srpt_system = QueueSystem([SRPT(sizefn = genExp(MU))], arrivalfn = genExp(LAMBDA))
```

Run the simulations and collect mean number of jobs and respone time metrics:
```python
N_fcfs,T_fcfs = fcfs_system.sim()
N_srpt,T_srpt = srpt_system.sim()
```

Plot them for a visual comparison:
```python
x_axis = ['FCFS', 'SRPT']
y_axis = [T_fcfs, T_srpt]
plt.bar(x_axis, y_axis, color='skyblue', edgecolor='black')
plt.ylabel('Mean Response Time')
plt.xlabel('Scheduling Policy')
plt.title('Comparison of FCFS vs SRPT')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

The result:

![FCFS vs SRPT response time in M/M/1](images/FCFSvsSRPT_MM1.png)

### Further Analysis
This is great and all, but we only experimented with one possible arrival rate, and thus server load. But what if traffic is light, and there aren't very many arrivals? Or what if traffic is extremely heavy?

So let's compare under multiple arrival rates:
```python
ARRIVAL_RATES = [1.0, 3.0, 5.0, 7.0, 10.0]
MU = 10.0

ratios = []

for l in ARRIVAL_RATES:
    fcfs_system = QueueSystem([FCFS(sizefn = genExp(MU))], arrivalfn = genExp(l))
    srpt_system = QueueSystem([SRPT(sizefn = genExp(MU))], arrivalfn = genExp(l))
    N_fcfs,T_fcfs = fcfs_system.sim()
    N_srpt,T_srpt = srpt_system.sim()
    # compare the ratio of response times
    ratios.append(T_fcfs/T_srpt)


plt.figure()
x_axis = ARRIVAL_RATES
y_axis = ratios
plt.plot(ARRIVAL_RATES, ratios, marker='o', linewidth=2, color='steelblue')
plt.xlabel('Arrival Rate λ')
plt.ylabel('E[T]_FCFS / E[T]_SRPT')
plt.title('Relative response time vs arrival rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```
![ratio of FCFS to SRPT response time in M/M/1](images/FCFSvsSRPTratio.png)

This shows us that at low traffic, the policies are similar. This makes sense, since when the server is mostly empty, the scheduling policy shouldn't have much of an effect (there isn't much queueing). However we see a spike in the ratio when the arrival rate is 10 jobs per sec, very close to the server speed of 12 jobs per sec, indicating that at high load, SRPT strongly outperforms FCFS.

### Summary

In this example, we saw how to interpret a problem as a queueing model, and compared two possible scheduling policies under different job arrival rates. 



