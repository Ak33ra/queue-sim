"""Example: static time-series plots for a 3-server network.

A non-Markovian routing network with heavy-tailed and uniform service.
Every server can route to every other with nonzero probability (plus
50% chance of exiting), so it is NOT a Jackson network.

Routing is symmetric:
    P(i -> j) = 0.25  for j != i
    P(exit)   = 0.50

Traffic equations give (with lambda_ext = 1.5):
    lambda_0 = 1.80,  rho_0 ~ 0.72  (fast Uniform server)
    lambda_1 = 0.60,  rho_1 ~ 0.63  (BoundedPareto — bursty)
    lambda_2 = 0.60,  rho_2 ~ 0.84  (wider Uniform — high variance)

Produces two plots:
  1. System state (total jobs in the network) over time
  2. Per-server occupancy heatmap
"""

from queue_sim import FCFS, QueueSystem, genBoundedPareto, genExp, genUniform
from queue_sim.plotting import plot_server_occupancy, plot_system_state

# --- Network setup ---
# Server 0: fast, low-variance (gateway server)       E[S] = 0.40
# Server 1: heavy-tailed BoundedPareto (bursty!)      E[S] = 1.06
# Server 2: wide Uniform (high variance, slower)      E[S] = 1.40
s0 = FCFS(sizefn=genUniform(0.1, 0.7))
s1 = FCFS(sizefn=genBoundedPareto(k=0.5, p=5.0, alpha=1.5))
s2 = FCFS(sizefn=genUniform(0.5, 2.3))

# Symmetric routing: 25% to each neighbor, 50% exit
M = [
    [0.0,  0.25, 0.25, 0.5],
    [0.25, 0.0,  0.25, 0.5],
    [0.25, 0.25, 0.0,  0.5],
]

system = QueueSystem([s0, s1, s2], arrivalfn=genExp(1.5), transitionMatrix=M)
system.sim(num_events=20_000, seed=42, track_events=True)

log = system.event_log
print(f"Logged {len(log)} events")

# --- Plot 1: system state over time ---
fig1, ax1 = plot_system_state(log)
ax1.set_title("Total jobs in system over time")
fig1.tight_layout()
fig1.savefig("system_state.png", dpi=150)
print("Saved system_state.png")

# --- Plot 2: per-server occupancy heatmap ---
fig2, ax2 = plot_server_occupancy(log, n_bins=400)
ax2.set_title("Per-server occupancy heatmap")
fig2.tight_layout()
fig2.savefig("server_occupancy.png", dpi=150)
print("Saved server_occupancy.png")
