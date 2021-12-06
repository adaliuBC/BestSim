import numpy as np
import matplotlib.pyplot as plt
import input
from input import const_input, section_input

'''
Izhikevich neuron model: 
'''
simu_time = 200. # simulation time in ms
ts = 0.01        # time step in ms

mode2param = {
    'tonic spiking':     [0.02, 0.40, -65.0, 2.0],
    'phasic spiking':    [0.02, 0.25, -65.0, 6.0],
    'tonic bursting':    [0.02, 0.20, -50.0, 2.0],
    'phasic bursting':   [0.02, 0.25, -55.0, 0.05]
}

class Izhikevich:
    '''
    Izhikevich neuron model: 
    '''
    def __init__(self, a = 0.02, b = 0.4, c = -65., d = 2., V_th = 30.):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V_th = V_th
    
    def simulate(self, t, u, V, spike, last_spike_t, stim, ts):
        spike = 0.
        dudt = self.a * (self.b * V - u)
        dVdt = 0.04*V*V + 5*V + 140 - u + stim
        u = u + ts * dudt
        V = V + ts * dVdt
        if V >= self.V_th:
            u = u + self.d
            V = self.c
            spike = 1.
            last_spike_t = t
        t += ts
        return t, u, V, spike, last_spike_t

def plot(mode, fig=None, ax1=None):
    # prepare lists to record the values
    t_list = []
    u_list = []
    V_list = []
    spike_list = []
    last_spike_t_list = []

    epochs = int(simu_time // ts + 1)   # num of stimulation epoch

    a    = mode2param[mode][0]
    b    = mode2param[mode][1]
    c    = mode2param[mode][2]
    d    = mode2param[mode][3]
    V_th = 30.              # in mV
    
    stim = mode2stim[mode]  # input list
    
    neu = Izhikevich(a = a, b = b, c = c, d = d, V_th = V_th)

    t = 0.
    u = 1.
    V = -65.
    spike = 0.
    last_spike_t = 1e-6
    for epoch in range(epochs):
        t_new = t + ts
        t_new, u_new, V_new, spike_new, last_spike_t_new = neu.simulate(
            t, u, V, spike, last_spike_t, stim[epoch], ts
        )

        t_list.append(t_new)
        u_list.append(t_new)
        V_list.append(V_new)
        spike_list.append(spike_new)
        last_spike_t_list.append(last_spike_t_new)
        t = t_new
        u = u_new
        V = V_new
        spike = spike_new
        last_spike_t = last_spike_t_new

    spike_list = np.array(spike_list)
    spike_cnt = np.sum(spike_list)
    '''ax1.set_title(mode)
    ax1.plot(t_list, V_list, 'steelblue', label='V')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane potential (mV)', color='steelblue')
    ax1.set_xlim(-0.1, 200.1)
    ax1.tick_params('y', colors='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(t_list, stim, 'coral', label='Input')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Input (mV)', color='coral')
    ax2.set_ylim(0, 100)
    ax2.tick_params('y', colors='coral')
    ax1.legend(loc=1)
    ax2.legend(loc=3)
    fig.tight_layout()'''
    return spike_cnt

mode = 'tonic bursting'
mode2stim = {}

# compare between inputs 
# (TODO: Must comment plot part in plot func when using this!)

range_start = 15
range_end = 105
fig, axs = plt.subplots(1, 1)
# test const input in a range
input_type = "constant input"
amp_list = []
const_cnt_list = []
for amp in range(range_start, range_end, 10):
    #fig, axs = plt.subplots(1, 1)
    stim = input.const_input(amp = amp, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot(mode)
    print(f"{input_type} generates {spike_cnt} spikes in {simu_time}ms.")
    amp_list.append(amp)
    const_cnt_list.append(spike_cnt)

# test ramp input in a range
input_type = "ramp input"
amp_list = []
ramp_cnt_list = []
for amp in range(range_start, range_end, 10):
    start_amp = amp - 5
    end_amp = amp + 5
    #fig, axs = plt.subplots(1, 1)
    stim = input.ramp_input(ampSt = start_amp, ampEnd = end_amp, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot(mode)
    print(f"{input_type} generates {spike_cnt} spikes in {simu_time}ms.")
    amp_list.append(amp)
    ramp_cnt_list.append(spike_cnt)
fig, axs = plt.subplots(1, 1)

# test sin input in a range
input_type = "sin input"
amp_list = []
sin_cnt_list = []
for amp in range(range_start, range_end, 10):
    start_amp = amp - 5
    end_amp = amp + 5
    #fig, axs = plt.subplots(1, 1)
    stim = input.sinuous_input(mean = amp, amp = 5., omega = np.pi/200., b = 0., 
                           simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot(mode)
    print(f"{input_type} generates {spike_cnt} spikes in {simu_time}ms.")
    amp_list.append(amp)
    sin_cnt_list.append(spike_cnt)

# test biphasic input in a range
input_type = "biphasic input"
amp_list = []
biphasic_cnt_list = []
for amp in range(range_start, range_end, 10):
    #fig, axs = plt.subplots(1, 1)
    stim = input.biphasic_input(mean = amp, amp = 5., period = 10., 
                               simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot(mode)
    print(f"{input_type} generates {spike_cnt} spikes in {simu_time}ms.")
    amp_list.append(amp)
    biphasic_cnt_list.append(spike_cnt)

# plot
# fmt = '[marker][line][color]'
fig, axs = plt.subplots(1, 1)
plt.plot(amp_list, const_cnt_list, '-b', label="const cnt-amp")
plt.plot(amp_list, ramp_cnt_list, '.-c', label="ramp cnt-amp")
plt.plot(amp_list, sin_cnt_list, 'x-g', label="sin cnt-amp")
plt.plot(amp_list, biphasic_cnt_list, '+-r', label="bipha cnt-amp")
plt.legend()
plt.show()



'''
# test the freq of sin and biphasic
input_type = "sinuous input"
period_list = []
sin_cnt_list = []
for omega in [50*np.pi, 25*np.pi, 10*np.pi, 5*np.pi, 1, 
              np.pi/5, np.pi/10, np.pi/25, np.pi/50,
              np.pi/100, np.pi/125, np.pi/150, np.pi/200]:
    #fig, axs = plt.subplots(1, 1)
    stim = input.sinuous_input(mean = 20., amp = 5., omega = omega, b = 0., 
                           simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    print(f"{input_type} generates {spike_cnt} spikes with omega={omega} in {simu_time}ms.")
    period = np.pi/omega
    period_list.append(period)
    sin_cnt_list.append(spike_cnt)

# plot
# fmt = '[marker][line][color]'
fig = plt.figure()
plt.plot(period_list, sin_cnt_list, '*-', label="Sinuous Input")
plt.legend()
#plt.show()

# TODO: test the freq of bipha
input_type = "biphasic input"
period_list = []
sin_cnt_list = []
for period in range(0, 200, 5):
    #fig, axs = plt.subplots(1, 1)
    stim = input.biphasic_input(mean=0., amp=20., period=period, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    print(f"{input_type} generates {spike_cnt} spikes with period={period} in {simu_time}ms.")
    period_list.append(period)
    sin_cnt_list.append(spike_cnt)

# plot
# fmt = '[marker][line][color]'
plt.plot(period_list, sin_cnt_list, '+-', label="Biphasic Input")
plt.xlabel('Period')
plt.ylabel('Number of Spikes')
plt.legend()
plt.show()
'''

'''
for alpha, beta and gamma oscillation
(meaning less for freq is too high for biphasic inputs)
# TODO: test the rate of sin
#alpha osc: 8-12Hz
#beta osc 12.5-30Hz
#gamma osc: 25-40Hz
freq_list_alpha = [8, 9, 10, 11, 12]
sin_cnt_list_alpha = []
bip_cnt_list_alpha = []
for freq in freq_list_alpha:
    # for sin input
    period = 1/freq
    omega = np.pi/period
    stim = input.sinuous_input(mean = 20., amp = 5., omega = omega, b = 0., 
                           simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    sin_cnt_list_alpha.append(spike_cnt)
    
    # for bip input
    stim = input.biphasic_input(mean=0., amp=20., period=period, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    bip_cnt_list_alpha.append(spike_cnt)

freq_list_beta = range(13, 30, 3)
sin_cnt_list_beta = []
bip_cnt_list_beta = []
for freq in freq_list_beta:
    # for sin input
    period = 1/freq
    omega = np.pi/period
    stim = input.sinuous_input(mean = 20., amp = 5., omega = omega, b = 0., 
                           simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    sin_cnt_list_beta.append(spike_cnt)
    
    # for bip input
    stim = input.biphasic_input(mean=0., amp=20., period=period, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    bip_cnt_list_beta.append(spike_cnt)

freq_list_gamma = range(25, 40, 5)
sin_cnt_list_gamma = []
bip_cnt_list_gamma = []
for freq in freq_list_gamma:
    # for sin input
    period = 1/freq
    omega = np.pi/period
    stim = input.sinuous_input(mean = 20., amp = 5., omega = omega, b = 0., 
                           simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    sin_cnt_list_gamma.append(spike_cnt)
    
    # for bip input
    stim = input.biphasic_input(mean=0., amp=20., period=period, simuTime=simu_time, step=ts)
    mode2stim[mode] = stim
    spike_cnt = plot("tonic spiking")
    bip_cnt_list_gamma.append(spike_cnt)

# plot
# fmt = '[marker][line][color]'
fig, ax = plt.subplots()
ax.plot(freq_list_alpha, sin_cnt_list_alpha, '+-', 
         color="steelblue", label="Sinuous Input")
ax.plot(freq_list_alpha, bip_cnt_list_alpha, '*-', 
         color="darkorange", label="Biphasic Input")
ax.plot(freq_list_beta, sin_cnt_list_beta, '+-', 
         color="steelblue", label="Sinuous Input")
ax.plot(freq_list_beta, bip_cnt_list_beta, '*-', 
         color="darkorange", label="Biphasic Input")
ax.plot(freq_list_gamma, sin_cnt_list_gamma, '+-', 
         color="steelblue", label="Sinuous Input")
ax.plot(freq_list_gamma, bip_cnt_list_gamma, '*-', 
         color="darkorange", label="Biphasic Input")
plt.legend()
ax.axvspan(8, 12, facecolor='mistyrose')
ax.axvspan(12.5, 30, facecolor='wheat')
ax.axvspan(25, 40, facecolor='paleturquoise')
plt.show()
'''