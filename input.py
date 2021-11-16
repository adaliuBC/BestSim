import numpy as np

def const_input(amp = 0., simuTime = 1000., step = 0.01):
    '''
    Param:
    amp:         amplitude of stimulus
    simuTime:    total time of simulation
    step:        simulation step
    
    Return:
    input:       np.ndarray of constant stimulus amplitude per timestep
    '''
    epoch = int(simuTime // step + 1)
    input = np.array([amp] * epoch)
    return input

def biphahsic_input(amp = 0.,
                    simuTime = 1000., step = 0.01): 
    '''
    Param:
    amp:         amplitude of biphasic stimulus
    simuTime:    total time of simulation
    step:        simulation step
    
    Return:
    input:       np.ndarray of constant stimulus amplitude per timestep
    '''
    epoch = int(simuTime // step + 1)
    input = np.array([amp] * (epoch//2) + [-amp] * (epoch - epoch//2))
    return input


def ramp_input(ampSt = 0., ampEnd = 1., 
               simuTime = 1000., step = 0.01):
    '''
    Param:
    ampSt:       Start value of ramping stim
    ampEnd:      End value of ramping stim
    simuTime:    total time of simulation
    step:        simulation step
    
    Return:
    input:       np.ndarray of constant stimulus amplitude per timestep
    '''
    epoch = int(simuTime // step + 1)
    inputList = []
    for i in range(epoch):
        inputList.append(ampSt + (ampEnd - ampSt) / epoch)
    input = np.array(inputList)
    return input

def sinuous_input(amp = 1., omega = 1., b = 0.,
                  simuTime = 1000., step = 0.01):
    '''
    Param:
    amp:         Max amplitude of sinuous stim
    omega:       omega of sinuous stim
    b:           b of sinuous stim
    simuTime:    total time of simulation
    step:        simulation step
    
    Return:
    input:       np.ndarray of constant stimulus amplitude per timestep
    '''
    epoch = int(simuTime // step + 1)
    inputList = []
    for i in range(epoch):
        t = i * step
        inputList.append(amp * np.sin(omega * t + b))
    input = np.array(inputList)
    return input
