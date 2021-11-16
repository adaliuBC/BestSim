import numpy as np
import copy
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

def LIF(V_rest = -70., V_reset = -70., V_th = -55., R = 1., 
        tau = 10., stim = None, simu_time = 100., ts = 0.01):
    # set init values
    t = 0.          # time
    V = -70         # membrane potential
    epoch = int(simu_time // ts + 1)   # num of stim epoch
    for i in range(epoch):
        dvdt = - ((V - (V_rest + stim[i] * R)) / tau)
        V = V + ts * dvdt
        if V >= V_th:
            V = V_reset
        t_list.append(t)
        V_list.append(V)
        t += ts

# record value in list
t_list = []
V_list = []
stim1 = const_input(15., simu_time, ts)
# list of input, change with different input waveform
LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R,
    tau = tau, stim = stim1, simu_time = simu_time, ts = ts)
V_list1 = copy.deepcopy(V_list)

# record value in list
t_list = []
V_list = []
stim2 = const_input(16., simu_time, ts)
# list of input, change with different input waveform
LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R,
    tau = tau, stim = stim2, simu_time = simu_time, ts = ts)
V_list2 = copy.deepcopy(V_list)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t_list, stim1, label = "input=15")
axs[0].plot(t_list, stim2, label = "input=16")
axs[0].set_ylim(0., 20.)
axs[1].plot(t_list, V_list1, label = 'V1')
axs[1].plot(t_list, V_list2, label = 'V2')
plt.legend()
plt.show()
