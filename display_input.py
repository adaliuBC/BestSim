from input import *
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
t_list = np.array(range(0, 20000)) * 0.01
const_input = const_input(amp = 10., simuTime=200, step=0.01)
#axs[0][0].set_title("Constant Input")
plt.plot(t_list, const_input, label="Constant Input")
#axs[0][0].set_xlabel('Time')
#axs[0][0].set_ylabel('Input Amplitude')

ramp_input = ramp_input(ampSt = 0., ampEnd = 20., simuTime=200, step=0.01)
#axs[0][1].set_title("Ramp Input")
plt.plot(t_list, ramp_input, label="Ramp Input")
#axs[0][1].set_xlabel('Time (ms)')
#axs[0][1].set_ylabel('Input Amplitude')

biphasic_input = biphasic_input(
    mean = 10., amp = 5., period = 10., simuTime=200, step=0.01)
#axs[1][0].set_title("Biphasic Input")
plt.plot(t_list, biphasic_input, label="Biphasic Input")
#axs[1][0].set_xlabel('Time (ms)')
#axs[1][0].set_ylabel('Input Amplitude')

sinuous_input = sinuous_input(
    mean = 10., amp = 5., omega = np.pi/10, b = 0.,
    simuTime=200, step=0.01)
#axs[1][1].set_title("Sinuous Input")
plt.plot(t_list, sinuous_input, label="Sinuous Input")
plt.xlabel('Time (ms)')
plt.ylabel('Input Amplitude')
plt.legend()
plt.show()




