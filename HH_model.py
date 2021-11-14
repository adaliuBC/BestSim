import numpy as np
import matplotlib.pyplot as plt
from input import const_input

'''
HH neuron model: 
'''

C      = 1.
g_Na   = 120.
g_K    = 36.
g_leak = 0.03
E_Na   = 50.
E_K    = -77.
E_leak = -54.387
V_th = 20.

simu_time = 100. # simulation time in ms
ts = 0.01        # time step in ms

epoch = int(simu_time // ts + 1)   # num of stim epoch

stim = const_input(5., simu_time, [[0., 100.]], ts)
# list of input, change with different input waveform

t = 0.
m = 0.5
h = 0.6
n = 0.32
V = 0.

# record value in list
t_list = []
m_list = []
h_list = []
n_list = []
V_list = []

for i in range(epoch):
    # compute pos value
    alpha_m = ( 0.1 * (V+40) )/(1 - np.exp( -(V+40)/10 ))
    beta_m = 4.0 * np.exp( -(V+65)/(18) )
    alpha_h = 0.07 * np.exp( -(V+65)/(20) )
    beta_h = 1 / (1 + np.exp( -(V+35)/(10) ))
    alpha_n = (0.01 * (V + 55)) / (1 - np.exp( -(V+55)/(10) ))
    beta_n = 0.125 * np.exp( -(V+65)/(80) )
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n
    dVdt = ( - (g_Na*(m**3)*h*(V-E_Na) \
                + g_K*(n**4)*(V-E_K) \
                + g_leak*(V-E_leak)) \
            + stim[i] ) / C

    # update
    m_new = m + ts * dmdt
    h_new = h + ts * dhdt
    n_new = n + ts * dndt
    V_new = V + ts * dVdt
    m = m_new
    h = h_new
    n = n_new
    V = V_new  
    ## use the param val of last time step

    # save
    t_list.append(t)
    m_list.append(m)
    h_list.append(h)
    n_list.append(n)
    V_list.append(V)

    # next step
    t += ts


fig, axs = plt.subplots(2, 2)
axs[0][0].plot(t_list, stim, label = "input")
axs[0][1].plot(t_list, V_list, label = 'V')

axs[1][0].plot(t_list, m_list, label = 'm')
axs[1][1].plot(t_list, n_list, label = 'n')
plt.legend()
plt.show()
