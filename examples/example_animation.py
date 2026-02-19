"""Example: animated network visualization for a 3-server network.

Tuned for visual clarity: low arrival rate + few departures so that
each animation frame covers only a handful of events.  You can see
individual jobs arrive at server 0, get routed to 1 or 2, and
eventually depart.

Server nodes are colored by occupancy (YlOrRd) and show the current
queue length as a number.  Color changes from white (empty) through
yellow/orange to red (busy).

Saves the animation as a GIF.  To display inline in Jupyter, just
call `anim` in the last cell instead of saving.
"""

from queue_sim import FCFS, QueueSystem, genExp, genUniform
from queue_sim.animate import animate_network

# --- Network setup ---
# Low arrival rate + moderate service → occupancy stays in 0-6 range,
# so each arriving/departing job causes a visible color change.
s0 = FCFS(sizefn=genUniform(0.3, 0.9))   # E[S]=0.6, fast gateway
s1 = FCFS(sizefn=genUniform(0.8, 2.0))   # E[S]=1.4, slower
s2 = FCFS(sizefn=genUniform(1.0, 2.2))   # E[S]=1.6, slowest

# Asymmetric routing so traffic is visible on all edges
M = [
    [0.0,  0.35, 0.25, 0.4],    # s0 → s1 often, s2 sometimes
    [0.15, 0.0,  0.30, 0.55],   # s1 → s2 often, feedback to s0
    [0.20, 0.10, 0.0,  0.70],   # s2 → s0 feedback, some to s1
]

# Only 500 departures → ~1500 total events → a few events per frame
system = QueueSystem([s0, s1, s2], arrivalfn=genExp(1.0), transitionMatrix=M)
system.sim(num_events=500, seed=42, track_events=True)

log = system.event_log
print(f"Logged {len(log)} events over t=[0, {log.times[-1]:.1f}]")

# --- Animate ---
positions = {0: (0.5, 0.9), 1: (0.15, 0.2), 2: (0.85, 0.2)}

anim = animate_network(
    log,
    transition_matrix=M,
    positions=positions,
    n_frames=300,      # more frames for smoother playback
    interval=80,       # 80ms between frames
    node_size=1400,
    title="3-Server Routing Network",
)

anim.save("network_animation.gif", writer="pillow", fps=12)
print("Saved network_animation.gif")
