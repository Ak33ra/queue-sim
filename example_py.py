from queue_sim import QueueSystem, FCFS, SRPT, PS, FB, genExp, genUniform

# --- Scheduling policies ---

# M/M/1-FCFS: Poisson arrivals (rate 1), exponential service (rate 2)
system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)
# E[T] = 1/(mu - lam) = 1.0

# M/M/1-SRPT: preempts current job when a shorter one arrives
system = QueueSystem([SRPT(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# M/G/1-PS: all jobs share the server equally (rate 1/n each)
system = QueueSystem([PS(sizefn=genUniform(0.3, 0.7))], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)
# E[T] = E[S] / (1 - rho) for any service distribution

# M/M/1-FB: always serves job(s) with least attained service
system = QueueSystem([FB(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# --- Multi-server (G/G/k) ---

# M/M/2-FCFS: 2 parallel servers, each with rate 1
system = QueueSystem([FCFS(sizefn=genExp(1.0), num_servers=2)], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)
# E[T] = 4/3 (Erlang-C formula)

# M/M/2-PS: 2 servers shared among all jobs
system = QueueSystem([PS(sizefn=genExp(1.0), num_servers=2)], arrivalfn=genExp(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# --- Finite buffers (loss queues) ---

# M/M/3/3 (Erlang-B): 3 servers, capacity 3 — no waiting room
server = FCFS(sizefn=genExp(1.0), num_servers=3, buffer_capacity=3)
system = QueueSystem([server], arrivalfn=genExp(2.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# M/M/1/5: single server, capacity 5
server = FCFS(sizefn=genExp(2.0), buffer_capacity=5)
system = QueueSystem([server], arrivalfn=genExp(1.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# --- Networks ---

# Tandem: FCFS -> SRPT (jobs flow through in series)
system = QueueSystem(
    [FCFS(sizefn=genExp(4.0)), SRPT(sizefn=genExp(4.0))],
    arrivalfn=genExp(1.0),
)
N, T = system.sim(num_events=10**6, seed=42)

# Feedback: 30% of jobs return to server 0 after completion
system = QueueSystem([PS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
system.updateTransitionMatrix([[0.3, 0.7]])  # [to server 0, exit]
N, T = system.sim(num_events=10**6, seed=42)

# --- Replications with confidence intervals ---

system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
result = system.replicate(
    n_replications=30, num_events=10**6, seed=42, warmup=10_000,
)
print(f"E[T] = {result.mean_T:.4f}  95% CI: {result.ci_T}")

# --- Response time distribution tracking ---

import numpy as np

system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
system.sim(num_events=10**6, seed=42, track_response_times=True)
rt = np.array(system.response_times)
print(f"Median: {np.median(rt):.4f}, P99: {np.percentile(rt, 99):.4f}")

# Works with any policy
system = QueueSystem([SRPT(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
system.sim(num_events=10**6, seed=42, track_response_times=True)
rt = np.array(system.response_times)
print(f"SRPT Median: {np.median(rt):.4f}, P99: {np.percentile(rt, 99):.4f}")

# --- Plotting (pip install queue-sim[viz]) ---

from queue_sim.plotting import plot_cdf, plot_tail, compare_policies

# CDF of response times
fig, ax = plot_cdf(system.response_times, label="SRPT")

# Tail probability P(T > t) on log scale
fig, ax = plot_tail(system.response_times, label="SRPT")

# Compare multiple policies side-by-side
policies = {}
for name, cls in [("FCFS", FCFS), ("SRPT", SRPT), ("PS", PS)]:
    sys = QueueSystem([cls(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
    sys.sim(num_events=10**6, seed=42, track_response_times=True)
    policies[name] = sys.response_times
fig, ax = compare_policies(policies, kind="cdf")
fig, ax = compare_policies(policies, kind="tail")

fig.savefig("cdf.png")