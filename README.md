# queue-sim

A discrete-event simulation engine for queueing networks, with a C++ hot-path backend exposed to Python via pybind11.

Supports pluggable scheduling policies (FCFS, SRPT, PS, FB), multi-server queues (G/G/k), finite-buffer loss queues (M/M/c/c Erlang-B, M/M/1/K), tandem and feedback networks with probabilistic routing, opt-in per-job response time distribution tracking, event logging with trajectory reconstruction and animated network visualization, and statistically rigorous output analysis via independent replications with confidence intervals.

## Architecture

```
queue_sim/          Python frontend — system construction, replication logic, statistics
csrc/               C++ backend — event loop, servers, distributions (pybind11)
tests/              268 tests — analytical validation, Little's law, property-based (Hypothesis)
```

**Dual backend.** The same `QueueSystem` interface is available in pure Python and as a compiled C++ extension. The C++ event loop releases the GIL during simulation, enabling concurrent execution.

**Server abstraction.** Scheduling policies inherit from an abstract `Server` base class and implement arrival/completion logic independently. Current policies:
- **FCFS** — first-come first-served (supports `num_servers` for G/G/k)
- **SRPT** — shortest remaining processing time (preemptive)
- **PS** — processor sharing (supports `num_servers` for G/G/k — all jobs share k servers)
- **FB** — foreground-background / least attained service (serves jobs with least accumulated service)

**Multi-server queues (G/G/k).** FCFS and PS accept a `num_servers` parameter. With k servers, FCFS runs up to k jobs in parallel (rest wait in FIFO queue); PS shares k servers among all n jobs (rate min(k,n)/n per job). Validated against the Erlang-C formula for M/M/k.

**Finite buffers + loss queues.** All policies accept a `buffer_capacity` parameter (total system capacity K = in-service + waiting). Arrivals to a full server are rejected. Per-server `num_rejected` and `num_arrivals` counters enable computing loss probability P(loss). Supports M/M/c/c (Erlang-B), M/M/1/K, and arbitrary finite-buffer configurations. Validated against the Erlang-B formula and the M/M/1/K analytical loss probability.

**Response time distributions.** Pass `track_response_times=True` to `sim()` to record every measurement-phase job's response time. The resulting `system.response_times` list feeds directly into numpy/matplotlib for CDFs, percentiles, histograms, and tail analysis. Disabled by default for zero overhead.

**Event logging.** Pass `track_events=True` to `sim()` to record every arrival, departure, route, and rejection with timestamps, source/destination server indices, and system state. The resulting `system.event_log` enables full trajectory reconstruction and visualization. Works with both Python and C++ backends.

**Visualization.** Built-in plotting and animation tools for event logs:
- `plot_system_state()` — step plot of total jobs in the network over time
- `plot_server_occupancy()` — time-series heatmap of per-server occupancy via `pcolormesh`
- `animate_network()` — animated network diagram with nodes colored by occupancy, directed routing edges, and per-node queue length labels; returns a `FuncAnimation` for saving as GIF/MP4 or inline Jupyter display

**Statistical output.** `replicate()` runs N independent replications with deterministic per-replication seeds (SplitMix64), optional warmup, and returns t-distribution confidence intervals — no scipy dependency.

## Installation

```bash
pip install "git+https://github.com/Ak33ra/queue-sim.git"
```

For development (editable install, compiles C++ extension):

```bash
git clone https://github.com/Ak33ra/queue-sim.git
cd queue-sim
pip install -e ".[dev]"
```

Requires a C++17 compiler and Python >= 3.9.

## Usage

Both backends expose the same `QueueSystem` interface with a few differences:

| | Python | C++ |
|---|---|---|
| **Distributions** | Any `Callable[[], float]` — custom distributions, mixtures, etc. | Three built-in types only (`ExponentialDist`, `UniformDist`, `BoundedParetoDist`) |
| **Multi-server (G/G/k)** | `num_servers` param on FCFS, PS | `num_servers` param on FCFS, PS |
| **Finite buffers** | `buffer_capacity` param on all policies (`None` = unlimited) | `buffer_capacity` param on all policies (`-1` = unlimited) |
| **Response time tracking** | `track_response_times=True` on `sim()` | `track_response_times=True` on `sim()` |
| **Event logging** | `track_events=True` on `sim()` | `track_events=True` on `sim()` |
| **Parallel replications** | Sequential only | `n_threads` parameter for multithreaded execution |
| **GIL** | Held during simulation | Released — won't block other Python threads |

### Python Backend

```python
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

# --- Event logging + trajectory visualization ---

from queue_sim import per_server_states
from queue_sim.plotting import plot_system_state, plot_server_occupancy
from queue_sim.animate import animate_network

# 3-server network with probabilistic routing
s0 = FCFS(sizefn=genUniform(0.1, 0.7))
s1 = FCFS(sizefn=genExp(2.0))
s2 = FCFS(sizefn=genUniform(0.5, 2.3))
M = [
    [0.0,  0.25, 0.25, 0.5],   # 25% to each neighbor, 50% exit
    [0.25, 0.0,  0.25, 0.5],
    [0.25, 0.25, 0.0,  0.5],
]
system = QueueSystem([s0, s1, s2], arrivalfn=genExp(1.5), transitionMatrix=M)
system.sim(num_events=20_000, seed=42, track_events=True)

log = system.event_log  # EventLog with times, kinds, from_servers, to_servers, states

# Reconstruct per-server occupancy
data = per_server_states(log)
# data["server_states"][s][i] = occupancy of server s after event i

# System state over time
fig, ax = plot_system_state(log)

# Per-server occupancy heatmap
fig, ax = plot_server_occupancy(log, n_bins=400)

# Animated network diagram (saves as GIF or display inline in Jupyter)
anim = animate_network(
    log,
    transition_matrix=M,
    positions={0: (0.5, 0.9), 1: (0.15, 0.2), 2: (0.85, 0.2)},
    n_frames=200,
    node_size=1200,
    title="3-Server Network",
)
anim.save("network.gif", writer="pillow", fps=12)
```

### C++ Backend

The C++ backend uses the same API but with distribution objects instead of callables. Simulations release the GIL, and `replicate()` supports parallel execution via `n_threads`.

```python
import _queue_sim_cpp as cpp

# --- Scheduling policies ---

# M/M/1-FCFS
system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# M/G/1-PS with uniform service
system = cpp.QueueSystem([cpp.PS(cpp.UniformDist(0.3, 0.7))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# M/M/1-FB
system = cpp.QueueSystem([cpp.FB(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
N, T = system.sim(num_events=10**6, seed=42)

# --- Multi-server (G/G/k) ---

# M/M/2-FCFS
system = cpp.QueueSystem(
    [cpp.FCFS(cpp.ExponentialDist(1.0), num_servers=2)], cpp.ExponentialDist(1.0)
)
N, T = system.sim(num_events=10**6, seed=42)

# M/M/2-PS
system = cpp.QueueSystem(
    [cpp.PS(cpp.ExponentialDist(1.0), num_servers=2)], cpp.ExponentialDist(1.0)
)
N, T = system.sim(num_events=10**6, seed=42)

# --- Finite buffers (loss queues) ---

# M/M/3/3 (Erlang-B)
server = cpp.FCFS(cpp.ExponentialDist(1.0), num_servers=3, buffer_capacity=3)
system = cpp.QueueSystem([server], cpp.ExponentialDist(2.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# M/M/1/5
server = cpp.FCFS(cpp.ExponentialDist(2.0), buffer_capacity=5)
system = cpp.QueueSystem([server], cpp.ExponentialDist(1.0))
system.sim(num_events=10**6, seed=42)
print(f"P(loss) = {server.num_rejected / server.num_arrivals:.4f}")

# --- Networks ---

# Tandem: PS -> FCFS
system = cpp.QueueSystem(
    [cpp.PS(cpp.ExponentialDist(4.0)), cpp.FCFS(cpp.ExponentialDist(4.0))],
    cpp.ExponentialDist(1.0),
)
N, T = system.sim(num_events=10**6, seed=42)

# --- Parallel replications ---

system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
raw = system.replicate(
    n_replications=30, num_events=10**6, seed=42, warmup=10_000, n_threads=4,
)
# raw.raw_T and raw.raw_N are lists of per-replication results

# Wrap with CI computation (no scipy needed)
from queue_sim.results import _build_replication_result
result = _build_replication_result(tuple(raw.raw_N), tuple(raw.raw_T), 0.95)
print(f"E[T] = {result.mean_T:.4f}  95% CI: {result.ci_T}")

# --- Response time distribution tracking ---

import numpy as np

system = cpp.QueueSystem([cpp.FCFS(cpp.ExponentialDist(2.0))], cpp.ExponentialDist(1.0))
system.sim(num_events=10**6, seed=42, track_response_times=True)
rt = np.array(system.response_times)
print(f"Median: {np.median(rt):.4f}, P99: {np.percentile(rt, 99):.4f}")
```

### Available Distributions

| Python | C++ | Parameters |
|---|---|---|
| `genExp(mu)` | `ExponentialDist(mu)` | rate `mu`, E[X] = 1/mu |
| `genUniform(a, b)` | `UniformDist(a, b)` | support [a, b] |
| `genBoundedPareto(k, p, alpha)` | `BoundedParetoDist(k, p, alpha)` | shape `alpha`, range [k, p] |

## Testing and Validation

```bash
pytest tests/ -v
```

Tests validate simulation output against closed-form results:

- **Analytical (M/M/1):** E[T] = 1/(mu - lambda), E[N] = rho/(1 - rho), verified for FCFS, PS, and FB within 5% tolerance
- **Analytical (M/G/1):** Pollaczek-Khinchine formula for FCFS, E[S]/(1-rho) for PS, with Uniform service
- **Analytical (M/M/k):** Erlang-C formula for FCFS and PS with k=2 servers, verified on both backends
- **Erlang-B (M/M/c/c):** loss probability matches recursive Erlang-B formula for multiple (lam, mu, c) configurations
- **M/M/1/K:** loss probability matches analytical formula for finite-buffer single-server queues
- **Little's Law:** E[N] = lambda * E[T] verified for both FCFS and SRPT
- **Response time tracking:** `len(response_times) == num_events`, all positive, `mean(response_times) ≈ E[T]` within 5%, deterministic, zero-impact when disabled; verified for all policies on both backends
- **Event logging:** parallel-vector consistency, non-decreasing times, departure/arrival/route/rejection semantics, per-server reconstruction invariant (sum of per-server pops = system state), non-negative occupancies with buffer rejections; verified on both backends
- **Visualization:** `plot_system_state`, `plot_server_occupancy`, `animate_network` return correct types, accept existing axes, produce expected plot elements
- **Confidence intervals:** 95% CI from `replicate()` covers the true E[T] on both Python and C++ backends
- **Property-based (Hypothesis):** fuzz tests for edge cases and invariant checking
- **Seed determinism:** identical seeds produce identical results; verified on both backends

## Examples

See `examples/` for worked examples:
- `example_FIFO_SRPT.py` — FCFS vs SRPT under varying load
- `example_MG1.py` — M/G/1 with a custom service distribution
- `example_timeseries.py` — system state and heatmap plots for a 3-server non-Markovian network
- `example_animation.py` — animated network visualization with routing and occupancy labels

At low load the policies perform similarly, but as utilization approaches 1, SRPT significantly outperforms FCFS in mean response time:

| | FCFS vs SRPT (rho -> 1) |
|---|---|
| ![FCFS vs SRPT response time](images/FCFSvsSRPT_MM1.png) | ![Ratio across arrival rates](images/FCFSvsSRPTratio.png) |

## Project Structure

```
queue_sim/
  __init__.py             Public API and exports
  queueSystem.py          QueueSystem — sim() and replicate()
  results.py              ReplicationResult, CI computation, seed derivation
  server.py               Abstract Server base class
  event_log.py            EventLog, per_server_states(), _bin_step_function()
  plotting.py             plot_cdf, plot_tail, compare_policies, plot_system_state, plot_server_occupancy
  animate.py              animate_network() — FuncAnimation for network state over time
  policies/
    FCFS.py               First-come first-served
    SRPT.py               Shortest remaining processing time
    PS.py                 Processor sharing
    FB.py                 Foreground-background (least attained service)
  lib/
    rvGen.py              Distribution samplers (Exp, Uniform, BoundedPareto)

csrc/
  include/queue_sim/      C++ headers (distributions, server, FCFS, SRPT, PS, FB, queue_system)
  src/bindings.cpp        pybind11 module definition

tests/                    268 tests (analytical, event log, visualization, animation, C++ backend)
examples/                 Worked examples (scheduling comparison, time-series plots, animation)
benchmarks/               Performance benchmarks
```

## License

MIT
