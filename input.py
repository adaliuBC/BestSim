import numpy as np

def const_input(amp = 0., simuTime = 1000., 
                stimPeriods = None, step = 0.01):
    '''
    Param:
    amp:         amplitude of stimulus
    simuTime:    total time of simulation
    simuPeriods: list of stimulus [startTime, endTime]
    step:        simulation step
    
    Return:
    input:       np.ndarray of constant stimulus amplitude per timestep
    '''
    epoch = int(simuTime // step + 1)
    input = np.array([0.] * epoch)
    for period in stimPeriods:
        start = int(period[0] // step)
        end = int(period[1] // step)
        input[start:end+1] = amp
    return input