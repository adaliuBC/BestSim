import numpy as np
import copy
import matplotlib.pyplot as plt
from input import const_input, ramp_input

'''
LIF neuron model: (dV/dt) = - (V - (V_rest + stim * R)) / tau
'''

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
    
    def simulate(self, t, V, spike, last_spike_t, stim, ts):
        spike = 0.
        dvdt = - ((V - (self.V_rest + stim * self.R)) / self.tau)
        V = V + ts * dvdt
        if V >= self.V_th:
            V = self.V_reset
            spike = 1.
            last_spike_t = t
        t += ts
        return t, V, spike, last_spike_t


if __name__ == "__main__":
    # LIF params
    V_rest = -70.   # resting potential in mV
    V_reset = -70.  # reset potential in mV
    V_th = -55.     # spike threshold potential in mV
    R = 1.          # membrane resistance in ?
    tau = 10.       # time const in ms

    # record value in list
    simu_time = 100 # simulation time in ms
    ts = 0.01       # time step in ms

    epochs = int(simu_time // ts + 1)
    stim_list = const_input(20., 100., ts)
    #stim_list = np.array([20.] * 1000 + [0.] * 9000)
    # list of input, change with different input waveform
    neu = LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R, tau = tau)

    t_list = []
    V_list = []
    spike_list = []
    last_spike_t_list = []

    t = 0.
    V = -70.
    spike = 0.
    last_spike_t = -1e6
    for epoch in range(epochs):
        t_new = t + ts
        _, V_new, spike_new, last_spike_t_new = neu.simulate(
            t, V, spike, last_spike_t, stim_list[epoch], ts
        )
        t_list.append(t_new)
        V_list.append(V_new)
        spike_list.append(spike_new)
        last_spike_t_list.append(last_spike_t_new)
        t = t_new
        V = V_new
        spike = spike_new
        last_spike_t = last_spike_t_new

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t_list, stim_list, label = f"input = 20")
    axs[1].plot(t_list, V_list, label = 'V1')
    plt.show()


    '''
    # compare with other type of inputs
    ## ramp input

    fig, axs = plt.subplots(2, 1)
    for startVal, endVal in [(10, 20)]:
        stim = ramp_input(startVal, endVal, simu_time, ts)
        t_list = []
        V_list = []
        LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R,
            tau = tau, stim = stim, simu_time = simu_time, ts = ts)
        axs[0].plot(t_list, stim, label = f"({startVal}, {endVal})")
        axs[1].plot(t_list, V_list, label = 'V1')
        plt.legend()
    plt.show()
        

    ## sinuous input
    fig, axs = plt.subplots(2, 1)
    for startVal, endVal in [(10, 20)]:
        stim = sinuous_input(amp, omega, b, simu_time, ts)
        t_list = []
        V_list = []
        LIF(V_rest = V_rest, V_reset = V_reset, V_th = V_th, R = R,
            tau = tau, stim = stim, simu_time = simu_time, ts = ts)
        axs[0].plot(t_list, stim, label = f"({startVal}, {endVal})")
        axs[1].plot(t_list, V_list, label = 'V1')
        plt.legend()
    plt.show()
    '''