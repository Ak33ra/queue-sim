from queue_sim import *
import matplotlib.pyplot as plt

LAMBDA = 10.0 # arrival rate in jobs per sec
MU = 12.0 # server speed in jobs per sec

''' --- FCFS --- '''
fcfs_system = QueueSystem([FCFS(sizefn = genExp(MU))], arrivalfn = genExp(LAMBDA))

''' --- SRPT ---'''
srpt_system = QueueSystem([SRPT(sizefn = genExp(MU))], arrivalfn = genExp(LAMBDA))

N_fcfs,T_fcfs = fcfs_system.sim()
N_srpt,T_srpt = srpt_system.sim()

x_axis = ['FCFS', 'SRPT']
y_axis = [T_fcfs, T_srpt]
plt.bar(x_axis, y_axis, color='skyblue', edgecolor='black')
plt.ylabel('Mean Response Time')
plt.xlabel('Scheduling Policy')
plt.title('Comparison of FCFS vs SRPT')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

'''
Experiment with multiple arrival rates
'''
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