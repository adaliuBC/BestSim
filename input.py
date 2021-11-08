import numpy as np

def const_input(amp = 0., simuTime = 1000., 
                simuPeriods = None, step = 0.01):
    epoch = int(simuTime // step + 1)
    input = np.array([0.] * epoch)
    for period in simuPeriods:
        start = int(period[0] // step)
        end = int(period[1] // step)
        input[start:end+1] = amp
    return input