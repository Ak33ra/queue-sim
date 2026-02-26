"""Compare E[T] for standard M/M/2-PS vs concave speedup (n^0.7).

Standard M/M/k-PS: total throughput = min(k, n).
Concave speedup:   total throughput = n^alpha  (alpha < 1).

With alpha=0.7 and k=2, concave speedup gives less throughput when n >= 2,
so E[T] should be higher.
"""

from queue_sim import PS, QueueSystem, genExp

MU = 4.0
K = 2
ALPHA = 0.7
NUM_EVENTS = 500_000
SEED = 42
LOADS = [0.3, 0.5, 0.7, 0.85, 0.95]

print(f"{'rho':>6}  {'lam':>6}  {'E[T] standard':>14}  {'E[T] concave':>13}  {'ratio':>6}")
print("-" * 60)

for rho in LOADS:
    lam = rho * K * MU  # rho = lam / (k * mu)

    sys_std = QueueSystem(
        [PS(sizefn=genExp(MU), num_servers=K)],
        arrivalfn=genExp(lam),
    )
    _N, T_std = sys_std.sim(num_events=NUM_EVENTS, seed=SEED)

    sys_conc = QueueSystem(
        [PS(sizefn=genExp(MU), num_servers=K, speed_fn=lambda n: n**ALPHA)],
        arrivalfn=genExp(lam),
    )
    _N, T_conc = sys_conc.sim(num_events=NUM_EVENTS, seed=SEED)

    print(f"{rho:6.2f}  {lam:6.2f}  {T_std:14.4f}  {T_conc:13.4f}  {T_conc / T_std:6.2f}")
