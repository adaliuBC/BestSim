import numpy as np
import copy
import matplotlib.pyplot as plt
from input import const_input
from LIF_model import LIF

class AMPA:
    '''
    AMPA synapse: (dg/dt) = alpha[T] * (1 - g) - beta * g
    post-syn current: I_syn = g_max * g * (V_post - E)
    '''
    def __init__(self, g_max = 0.42, E=0., alpha=0.98, beta=0.18, 
                 T=0.5, T_duration=0.5):
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

    def simulate(self, t, g, V_pre, spike_pre, last_spike_t_pre, 
                 V_post, ts):
        posT = self.T if ((t - last_spike_t_pre) < self.T_duration) else 0.
        dgdt = self.alpha * posT * (1 - g) - self.beta * g
        g = g + ts * dgdt
        stim_post = - self.g_max * g * (V_post - self.E)
        t += ts
        return t, g, stim_post

# LIF params
V_rest = -70.   # resting potential in mV
V_reset = -70.  # reset potential in mV
V_th = -55.     # spike threshold potential in mV
R = 1.          # membrane resistance in ?
tau = 10.       # time const in ms

# AMPA params
g_max = 0.42
E=0.
alpha=0.98
beta=0.18
T=0.5
T_duration=0.5

simu_time = 100 # simulation time in ms
ts = 0.01       # time step in ms

epochs = int(simu_time // ts + 1)
stim_pre_list = const_input(20., simu_time, ts)
stim_post_base_list = np.array([0] * epochs)
# list of input, change with different input waveform
neu_pre = LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R, tau = tau)
syn = AMPA(g_max = g_max, E = E, alpha = alpha, beta = beta, T = T, T_duration = T_duration)
neu_post = LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R, tau = tau)

t_list = []
V_pre_list = []
spike_pre_list = []
last_spike_t_pre_list = []
V_post_list = []
spike_post_list = []
last_spike_t_post_list = []
stim_post_list = []
g_syn_list = []


t = 0.
V_pre = -70.
spike_pre = 0.
last_spike_t_pre = -1e6
V_post = -70.
spike_post = 0.
last_spike_t_post = -1e6
g_syn = 0.
for epoch in range(epochs):
    t_new = t + ts
    _, V_pre_new, spike_pre_new, last_spike_t_pre_new = neu_pre.simulate(
        t, V_pre, spike_pre, last_spike_t_pre, stim_pre_list[epoch], ts
    )
    _, g_syn_new, stim_post = syn.simulate(
        t, g_syn, V_pre, spike_pre, last_spike_t_pre, V_post, ts
    )
    stim_post = stim_post_base_list[epoch] + stim_post
    _, V_post_new, spike_post_new, last_spike_t_post_new = neu_post.simulate(
        t, V_post, spike_post, last_spike_t_post, stim_post, ts
    )
    t_list.append(t_new)
    V_pre_list.append(V_pre_new)
    spike_pre_list.append(spike_pre_new)
    last_spike_t_pre_list.append(last_spike_t_pre_new)
    g_syn_list.append(g_syn_new)
    V_post_list.append(V_post_new)
    spike_post_list.append(spike_post_new)
    last_spike_t_post_list.append(last_spike_t_post_new)
    stim_post_list.append(stim_post)
    t = t_new
    V_pre = V_pre_new
    spike_pre = spike_pre_new
    last_spike_t_pre = last_spike_t_pre_new
    g_syn = g_syn_new
    V_post = V_post_new
    spike_post = spike_post_new
    last_spike_t_post = last_spike_t_post_new


fig, axs = plt.subplots(3, 2)
axs[0][0].plot(t_list, stim_pre_list, label = "input=20")
axs[1][0].plot(t_list, V_pre_list, label = 'V_pre')
axs[2][0].plot(t_list, spike_pre_list, label = 'spike_pre')

axs[0][1].plot(t_list, g_syn_list, label = "g_syn")
axs[1][1].plot(t_list, V_post_list, label = 'V_post')
axs[2][1].plot(t_list, stim_post_list, label = 'stim_post')
plt.legend()
plt.show()
