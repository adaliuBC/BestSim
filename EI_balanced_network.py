import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from input import const_input

class LIF:
    '''
    LIF neuron model: (dV/dt) = - (V - (V_rest + stim * R)) / tau
    '''
    def __init__(self, V_rest = -70., V_reset = -70., V_th = -55., R = 1., tau = 10.):
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
    
    def simulate(self, t, V, spike, stim, ts):
        #print(stim)
        n = len(V)  # num of neurons
        spike = np.array([0.] * n)
        #print(V.shape, spike.shape, stim.shape)
        dvdt = - ((V - (self.V_rest + stim * self.R)) / self.tau)
        V = V + ts * dvdt
        spike[V >= self.V_th] = 1.
        V[V >= self.V_th] = self.V_reset
        t += ts
        return t, V, spike

class exponential:  # size: (500*0.1) x (500*0.1)
    '''
    exponential synapse: (ds/dt) = - s / tau
    post-syn current: I_syn = g_max * s
    '''
    def __init__(self, g_max = 0.5, tau = 2.):
        self.g_max = g_max
        self.tau = tau

    def simulate(self, t, s, spike_pre, ts):
        # spike_pre: pre_num * 1
        # s: pre_num * post_num
        # stim_post: post_num * 1
        # print("exp: ", s.shape, spike_pre.shape)
        dsdt = - s / self.tau
        s += ts * dsdt
        s[spike_pre == 1.] += 1
        stim_post = self.g_max * np.dot(spike_pre, s)
        t += ts
        return t, s, stim_post

excit_neu_num = 500
inhib_neu_num = 500
conn_prob = 0.1

# LIF params
V_rest = -52.   # resting potential in mV
V_reset = -60.  # reset potential in mV
V_th = -50.     # spike threshold potential in mV
R = 1.          # membrane resistance in ?
tau = 10.       # time const in ms

# exponential params
g_max_E = 1 / (np.sqrt(conn_prob * excit_neu_num))
g_max_I = 1 / (np.sqrt(conn_prob * inhib_neu_num))
tau_decay = 2.

simu_time = 1000. # simulation time in ms
ts = 0.05        # time step in ms

epochs = int(simu_time // ts + 1)
stim_ext2E_list = np.ones((epochs, excit_neu_num)) * 3
stim_ext2I_list = np.ones((epochs, inhib_neu_num)) * 3
# list of input, change with different input waveform
neu_E = LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R, tau = tau)
neu_I = LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R, tau = tau)
syn_E2E = exponential(g_max = g_max_E, tau = tau_decay)
syn_E2I = exponential(g_max = g_max_E, tau = tau_decay)
syn_I2E = exponential(g_max = - g_max_I, tau = tau_decay)
syn_I2I = exponential(g_max = - g_max_I, tau = tau_decay)

# prepare to record t
# element as integer
t_list = []                
# prepare to record E neu
# element as ndarray of 1*excit_neu_num
V_E_list = []              
spike_E_list = []          
#last_spike_t_E_list = []
# prepare to record I neu
# element as ndarray of 1*inhib_neu_num
V_I_list = []              
spike_I_list = []
#last_spike_t_I_list = []
# prepare to record syn s
# element as ndarray of post_num * pre_num
s_syn_E2E_list = []        
s_syn_E2I_list = []
s_syn_I2E_list = []
s_syn_I2I_list = []

# set init value
# for t
t = 0.
# for E neus
V_E            = np.array(V_rest + np.random.random(excit_neu_num) * (V_th - V_rest))
spike_E        = np.array([0.] * excit_neu_num)
#last_spike_t_E = np.array([-1e6] * excit_neu_num)
# for I neus
V_I            = np.array(V_rest + np.random.random(inhib_neu_num) * (V_th - V_rest))
spike_I        = np.array([0.] * inhib_neu_num)
#last_spike_t_I = np.array([-1e6] * inhib_neu_num)
# for syns
s_syn_E2E = np.zeros((excit_neu_num, int(excit_neu_num * conn_prob)))
s_syn_E2I = np.zeros((inhib_neu_num, int(excit_neu_num * conn_prob)))
s_syn_I2E = np.zeros((excit_neu_num, int(inhib_neu_num * conn_prob)))
s_syn_I2I = np.zeros((inhib_neu_num, int(inhib_neu_num * conn_prob)))
# each row refers to a post neuron
#print("syn_E2E_size: ", s_syn_E2E.shape)

# build conn for E2E and I2E syns
conn_E2E_ind = []
conn_I2E_ind = []
conn_E2E_mat = []
conn_I2E_mat = []
for i in range(excit_neu_num):
    conn_E2pos_ind = random.sample(
        range(0, excit_neu_num), k = int(excit_neu_num * conn_prob)
    )
    conn_E2pos_filter = np.array([False] * excit_neu_num)
    for pre_ind in conn_E2pos_ind:
        conn_E2pos_filter[pre_ind] = True
    conn_E2E_mat.append(conn_E2pos_filter)
    
    conn_I2pos_ind = random.sample(
        range(0, inhib_neu_num), k = int(inhib_neu_num * conn_prob)
    )
    conn_I2pos_filter = np.array([False] * excit_neu_num)
    for pre_ind in conn_I2pos_ind:
        conn_I2pos_filter[pre_ind] = True
    conn_I2E_mat.append(conn_I2pos_filter)
conn_E2E_mat = np.array(conn_E2E_mat)
conn_I2E_mat = np.array(conn_I2E_mat)

# build conn for E2I and I2I syns
conn_E2I_ind = []
conn_I2I_ind = []
conn_E2I_mat = []
conn_I2I_mat = []
for i in range(inhib_neu_num):
    conn_E2pos_ind = random.sample(
        range(0, excit_neu_num), k = int(excit_neu_num * conn_prob)
    )
    conn_E2pos_filter = np.array([False] * inhib_neu_num)
    for pre_ind in conn_E2pos_ind:
        conn_E2pos_filter[pre_ind] = True
    conn_E2I_mat.append(conn_E2pos_filter)
    
    conn_I2pos_ind = random.sample(
        range(0, inhib_neu_num), k = int(inhib_neu_num * conn_prob)
    )
    conn_I2pos_filter = np.array([False] * inhib_neu_num)
    for pre_ind in conn_I2pos_ind:
        conn_I2pos_filter[pre_ind] = True
    conn_I2I_mat.append(conn_I2pos_filter)
conn_E2I_mat = np.array(conn_E2I_mat)
conn_I2I_mat = np.array(conn_I2I_mat)
# each row refers to all connections on one post neuron

start_t = time.time()
# simulate
for epoch in range(epochs):
    t_new = t + ts

    # get stim for each excit neuron
    for i in range(excit_neu_num):
        # E2E stim
        conn_E2pos_filter = conn_E2E_mat[i]
        _, s_syn_E2E_new, stim_E2pos = syn_E2E.simulate(
            t, s_syn_E2E[i], spike_E[conn_E2pos_filter], ts
        )
        # I2E stim
        conn_I2pos_filter = conn_I2E_mat[i]
        _, s_syn_I2E_new, stim_I2pos = syn_I2E.simulate(
            t, s_syn_I2E[i], spike_I[conn_I2pos_filter], ts
        )
        # add up all inputs
        stim_ext2E_list[epoch][i] += stim_E2pos + stim_I2pos
        # update E2E, I2E syns
        s_syn_E2E[i] = s_syn_E2E_new
        s_syn_I2E[i] = s_syn_I2E_new

    # get stim for each excit neuron
    for i in range(inhib_neu_num):
        # E2I stim
        conn_E2pos_filter = conn_E2I_mat[i]
        _, s_syn_E2I_new, stim_E2pos = syn_E2I.simulate(
            t, s_syn_E2I[i], spike_E[conn_E2pos_filter], ts
        )
        # I2I stim
        conn_I2pos_filter = conn_I2I_mat[i]
        _, s_syn_I2I_new, stim_I2pos = syn_I2I.simulate(
            t, s_syn_I2I[i], spike_I[conn_I2pos_filter], ts
        )
        # add up all inputs
        stim_ext2I_list[epoch][i] += stim_E2pos + stim_I2pos

        # update E2I, I2I syns
        s_syn_E2I[i] = s_syn_E2I_new
        s_syn_I2I[i] = s_syn_I2I_new
    
    
    # simu E neurons
    _, V_E_new, spike_E_new = neu_E.simulate(
        t, V_E, spike_E, stim_ext2E_list[epoch], ts
    )
    # simu I neurons
    _, V_I_new, spike_I_new = neu_I.simulate(
        t, V_I, spike_I, stim_ext2I_list[epoch], ts
    )

    # save time
    t_list.append(t_new)
    # save E neus
    V_E_list.append(V_E_new)
    spike_E_list.append(spike_E_new)
    #last_spike_t_E_list.append(last_spike_t_E_new)
    # save I neus
    V_I_list.append(V_I_new)
    spike_I_list.append(spike_I_new)
    #last_spike_t_I_list.append(last_spike_t_I_new)
    # save syns
    s_syn_E2E_list.append(s_syn_E2E)
    s_syn_E2I_list.append(s_syn_E2I)
    s_syn_I2E_list.append(s_syn_I2E)
    s_syn_I2I_list.append(s_syn_I2I)

    # update time
    t = t_new
    # update E neu
    V_E = V_E_new
    spike_E = spike_E_new
    #last_spike_t_E = last_spike_t_E_new
    # update I neu
    V_I = V_I_new
    spike_I = spike_I_new
    #last_spike_t_I = last_spike_t_I_new

    percent1 = int(epochs * 0.01)
    if epoch / percent1 == epoch // percent1:
        print(f"{epoch}/{epochs} finished, used {time.time() - start_t}s")

V_E_list = np.array(V_E_list)
V_I_list = np.array(V_I_list)
spike_E_list = np.array(spike_E_list)
spike_I_list = np.array(spike_I_list)
#print(spike_E_list.shape, spike_I_list.shape)

row_num = 4
col_num = 1
row_len = 2
col_len = 10
fig = plt.figure(figsize=(col_num * col_len, row_num * row_len), constrained_layout=True)
gs = GridSpec(row_num, col_num, figure=fig)
fig.add_subplot(gs[:3, 0])
spike_points_t = []
spike_points_n = []
for epoch in range(epochs):
    t = t_list[epoch]
    for i in range(excit_neu_num):
        if spike_E_list[epoch][i] == 1.:
            spike_points_t.append(t)
            spike_points_n.append(i)
plt.scatter(spike_points_t, spike_points_n, marker=".", linewidth=2)

def firing_rate(sp_matrix, width):
  rate = np.sum(sp_matrix, axis=1) / sp_matrix.shape[1]
  width1 = int(width / 2 / ts) * 2 + 1
  window = np.ones(width1) * 1000 / width
  return np.convolve(rate, window, mode='same')
'''
spike_sum = np.sum(spike_E_list, axis=1)
spike_sum = spike_sum / excit_neu_num
'''
rate = firing_rate(spike_E_list, 5.)
fig.add_subplot(gs[3, 0])
plt.plot(t_list, rate)
plt.show()
'''
axs[2].plot(t_list, V_E_list[:, 0])
plt.legend()
plt.show()
'''
