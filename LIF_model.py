import numpy as np
import matplotlib.pyplot as plt

'''
LIF neuron model: (dV/dt) = - (V - (V_rest + stim * R)) / tau
'''

V_rest = -70.   # resting potential in mV
V_reset = -70.  # reset potential in mV
V_th = -55.     # spike threshold potential in mV
R = 1.          # membrane resistance in ?
tau = 10.       # time const in ms

simu_time = 100 # simulation time in ms
ts = 0.01       # time step in ms

epoch = int(simu_time // ts)   # num of stim epoch

stim = [20.] * epoch           # list of input, change with different input waveform

t = 0.          # time
V = -70         # membrane potential
# record value in list
t_list = [t]
V_list = [V]

for i in range(epoch):
    dvdt = - ((V - (V_rest + stim[i] * R)) / tau)
    V = V + ts * dvdt
    if V >= V_th:
        V = V_reset
    t += ts
    t_list.append(t)
    V_list.append(V)

plt.plot(t_list, V_list)
#plt.legend()
plt.show()
