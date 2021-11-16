import numpy as np
import matplotlib.pyplot as plt
from input import const_input

'''
HH neuron model: 
'''
# tonic spiking set of params
a    = 0.02
b    = 0.4
c    = -65.
d    = 2.
V_th = 30.       # in mV

simu_time = 100. # simulation time in ms
ts = 0.01        # time step in ms

epoch = int(simu_time // ts + 1)   # num of stim epoch

stim = const_input(20., simu_time, ts)
# list of input, change with different input waveform

t = 0.
u = 1.
V = -65.

# record value in list
t_list = []
u_list = []
V_list = []

for i in range(epoch):
    # compute pos value
    dudt = a * (b * V - u)
    dVdt = 0.04*(V**2) + 5*V + 140 - u + stim[i]

    # update
    u_new = u + ts * dudt
    V_new = V + ts * dVdt
    if V_new > V_th:
        u_new += d
        V_new = c
    u = u_new
    V = V_new  
    ## use the param val of last time step

    # save
    t_list.append(t)
    u_list.append(u)
    V_list.append(V)

    # next step
    t += ts

fig, axs = plt.subplots(3, 1)
axs[0].plot(t_list, stim, label = "input")
axs[1].plot(t_list, u_list, label = 'u')
axs[2].plot(u_list, V_list, label = 'V')
plt.legend()
plt.show()
