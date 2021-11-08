import numpy as np
import matplotlib.pyplot as plt
from input import const_input

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

epoch = int(simu_time // ts + 1)   # num of stim epoch

stim = const_input(20., simu_time, [[20., 60.]], ts)
# list of input, change with different input waveform

t = 0.          # time
V = -70         # membrane potential
# record value in list
t_list = []
V_list = []

for i in range(epoch):
    dvdt = - ((V - (V_rest + stim[i] * R)) / tau)
    V = V + ts * dvdt
    if V >= V_th:
        V = V_reset
    t += ts
    t_list.append(t)
    V_list.append(V)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t_list, stim, label = "input")
axs[1].plot(t_list, V_list, label = 'V')
plt.legend()
plt.show()
