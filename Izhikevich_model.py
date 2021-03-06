import numpy as np
import matplotlib.pyplot as plt
import input
from input import const_input, section_input

'''
Izhikevich neuron model: 
'''
simu_time = 200. # simulation time in ms
ts = 0.01        # time step in ms

mode_list = ['tonic spiking', 'phasic spiking', 'tonic bursting', 'phasic bursting',   
             'mixed mode', 'spike frequency adaptation', 'class 1', 'class 2',
             'spike latency', 'subthreshold oscillations', 'resonator', 'integrator', 
             'rebound spike', 'threshold variability', 'bistability',
             'depolarizing afterpotential', 'rebound burst', 'accommodation', 'inhibition-induced spiking', 'inhibition-induced bursting']
mode2param = {
    'tonic spiking':              [0.02, 0.40, -65.0, 2.0],
    'phasic spiking':             [0.02, 0.25, -65.0, 6.0],
    'tonic bursting':             [0.02, 0.20, -50.0, 2.0],
    'phasic bursting':            [0.02, 0.25, -55.0, 0.05],
    'mixed mode':                 [0.02, 0.20, -55.0, 4.0],
    'spike frequency adaptation': [0.01, 0.20, -65.0, 8.0],
    'class 1':                    [0.02, -0.1, -55.0, 6.0],
    'class 2':                    [0.20, 0.26, -65.0, 0.0],
    'spike latency':              [0.02, 0.20, -65.0, 6.0],
    'subthreshold oscillations':  [0.05, 0.26, -60.0, 0.0],
    'resonator':                  [0.10, 0.26, -60.0, -1.0],
    'integrator':                 [0.02, -0.1, -55.0, 6.0],
    'rebound spike':              [0.03, 0.25, -60.0, 4.0],
    'rebound burst':              [0.03, 0.25, -52.0, 0.0],
    'threshold variability':      [0.03, 0.25, -60.0, 4.0],
    'bistability':                [1.00, 1.50, -60.0, 0.0],
    'depolarizing afterpotential':[1.00, 0.20, -60.0, -21.0],
    'accommodation':              [0.02, 1.00, -55.0, 4.0],
    'inhibition-induced spiking': [-0.02, -1.00, -60.0, 8.0],
    'inhibition-induced bursting':[-0.026, -1.00, -45.0, 0],

    'Regular Spiking':            [0.02, 0.2, -65, 8],
    'Intrinsically Bursting':     [0.02, 0.2, -55, 4],
    'Chattering':                 [0.02, 0.2, -50, 2],
    'Fast Spiking':               [0.1, 0.2, -65, 2],
    'Thalamo-cortical':           [0.02, 0.25, -65, 0.05],
    'Resonator':                  [0.1, 0.26, -65, 2],
    'Low-threshold Spiking':      [0.02, 0.25, -65, 2],
}

mode2stim = {}
mode = 'tonic spiking'
stim = input.section_input(20., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'phasic spiking'
stim = input.section_input(1., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'tonic bursting'
stim = input.section_input(15., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'phasic bursting'
stim = input.section_input(1., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'mixed mode'
stim = input.section_input(10., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'spike frequency adaptation'
stim = input.section_input(30., 50., 200., simu_time, ts)
mode2stim[mode] = stim
mode = 'class 1'
stim = input.ramp_input(0., 80., simu_time, ts)
mode2stim[mode] = stim
mode = 'class 2'
stim = input.ramp_input(0., 10., simu_time, ts)
mode2stim[mode] = stim
mode = 'spike latency'
stim = input.section_input(50., 15., 16., simu_time, ts)
mode2stim[mode] = stim
mode = 'subthreshold oscillations'
stim = input.section_input(50., 15., 16., 200., ts)
mode2stim[mode] = stim

mode = 'resonator'  #???
epoch = int(200. // ts + 1)
ampList = [-1, 0., -1, 0, -1, 0, -1, 0]
timeList = [10, 10, 10, 10, 100, 10, 30, 20]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'integrator'
epoch = int(200. // ts + 1)
ampList = [0, 48, 0, 48, 0, 48, 0, 48, 0]
timeList = [19, 1, 1, 1, 28, 1, 1, 1, 56]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'rebound spike'   #???
epoch = int(200. // ts + 1)
ampList = [7, 0, 7]
timeList = [10, 5, 185]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'rebound burst'   #???


mode = 'threshold variability'
epoch = int(200. // ts + 1)
ampList = [0, 5, 0, -5, 0, 5, 0]
timeList = [13, 3, 78, 2, 2, 3, 13]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'bistability'  #???
epoch = int(200. // ts + 1)
ampList = [0., 5., 0, 5, 0]
timeList = [10, 1, 10, 1, 10]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'depolarizing afterpotential'  #???
epoch = int(200. // ts + 1)
ampList = [0, 23, 0]
timeList = [7, 1, 50]
stim = []
posEpoch = 0
for i in range(len(ampList)):
    stim += [ampList[i]] * int(timeList[i] // ts + 1)
    posEpoch += int(timeList[i] // ts + 1)
stim += [0] * (epoch - posEpoch)
stim = np.array(stim)
mode2stim[mode] = stim

mode = 'accommodation'
mode = 'inhibition-induced spiking'
mode = 'inhibition-induced bursting'


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

def plot(mode, fig, ax1):
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

    ax1.set_title(mode)
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
    fig.tight_layout()

modeNum = 0
for i in range(5):
    fig, axs = plt.subplots(2, 2)
    plot(mode_list[modeNum], fig, axs[0][0])
    modeNum += 1
    plot(mode_list[modeNum], fig, axs[0][1])
    modeNum += 1
    plot(mode_list[modeNum], fig, axs[1][0])
    modeNum += 1
    plot(mode_list[modeNum], fig, axs[1][1])
    modeNum += 1
    plt.show()

