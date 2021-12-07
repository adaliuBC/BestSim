# Our source code for reproduction of the Golden paper
# Rhys Tracy and Xinyu Liu


import numpy as np
import math
import cv2
import pickle

from sklearn.linear_model import LinearRegression, Ridge

from matplotlib import pyplot as plt


# Function definitions -----------------------------------------------------------------------------------------------------------------------------------

#Function Definitions

def DoG_on_parasol_bipolars(r, amp):
    a_cent = 1
    a_sur = 0.01
    sig_cent = 4
    sig_sur = sig_cent * 9
    DoG = a_cent * math.e**(-(r**2)/(2*sig_cent**2)) - a_sur * math.e**(-(r**2)/(2*sig_sur**2))
    return DoG*amp

def DoG_on_parasol(r, amp):
    a_cent = 1
    a_sur = 0.375
    sig_cent = 10
    sig_sur = sig_cent * 1.15
    DoG = a_cent * math.e**(-(r**2)/(2*sig_cent**2)) - a_sur * math.e**(-(r**2)/(2*sig_sur**2))
    return DoG*amp

def simulate(cone_array):
    
    #RGC matrices
    on_parasol_responses = np.zeros((64,64))


    #Bipolar matrices
    on_parasol_bipolar_responses = np.zeros((256,256))


    #Cone matrices
    #cone_responses = np.zeros((256,256))
    cone_responses = np.copy(cone_array)
    
    # get bipolar responses from cone outputs

    # scale up by 11 (10 microns each axis = 5 neurons each way, this means we have 11x11 cones feeding into each bipolar
    # including the cone directly on top of the bipolar)

    i=0
    j=0
    while(i<np.shape(on_parasol_bipolar_responses)[0]):
        while(j<np.shape(on_parasol_bipolar_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 11
            b_max = 11

            if(j<5):
                b_init=5-j #only can go over to 0th col of cones
            if(i<5):
                a_init=5-i #only can go over to 0th row of cones
            if(j>250):
                b_max=6+(255-j) #only can go over to 255th col of cones
            if(i>250):
                a_max=6+(255-i) #only can go over to 255th row of cones

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):
                    #scale up and get radii:
                    r = math.sqrt(((a-5)**2) + ((b-5)**2))

                    #get scaled up response:
                    in_resp = cone_responses[i+(a-5),j+(b-5)]
                    resp = DoG_on_parasol_bipolars(r, in_resp)

                    #scale back down by summing:
                    on_parasol_bipolar_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    #temporal response


    #Since golden paper didn't give any information on the temporal filter (and we are using slightly different
    #temporal modeling with single frame of picture shown then nothing for the rest of the 0.4 s), we will use
    #exponential decay to model the temporal behavior with the initial value starting at the spacial response
    #and the decay value to be 50 (0.02s half life of neuron response strength, close to curve of golden
    #temporal response)

    #response = init_response * e^(decay_val*t)

    decay_val = -50


    #on parasol temporal:

    initial_on_parasol_bipolar_responses = np.copy(on_parasol_bipolar_responses)

    t=0.001 #time steps of 0.001s (400 time periods)
    while t<0.4: #simulate 0.4s like golden paper did
        i=0
        j=0
        while(i<np.shape(on_parasol_bipolar_responses)[0]):
            while(j<np.shape(on_parasol_bipolar_responses)[1]):
                #calculate current time's temporal response value with exponential decay equation, then add it
                #to the overall bipolar response

                init_response = initial_on_parasol_bipolar_responses[i,j]

                response = init_response * math.e**(decay_val*t)

                on_parasol_bipolar_responses[i,j] += response #sum over temporal response for each neuron

                j+=1
            i+=1

        t+=0.001
    
    # get rgc responses from bipolars

    # scale up by 32 from rgc matrix (need 25 microns/axis ~12 neurons each way, means we have 25x25 bipolars feeding each
    # rgc including the bipolar directly on top of the rgc)
    # we need our scaling factor to be a multiple of 4 though (which ignores symmetry), so we will scale by 32x32
    # so that despite assymetry, the value we are losing are essentially 0
    
    i=0
    j=0
    while(i<np.shape(on_parasol_responses)[0]):
        while(j<np.shape(on_parasol_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j*4<16):
                b_init=16-j*4 #only can go over to 0th col of bipolars
            if(i*4<16):
                a_init=16-i*4 #only can go over to 0th row of bipolars
            if(j*4>239):
                b_max=16+(255-j*4) #only can go over to 255th col of bipolars
            if(i*4>239):
                a_max=16+(255-i*4) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):

                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = on_parasol_bipolar_responses[i*4+(a-16),j*4+(b-16)]
                    resp = DoG_on_parasol(r, in_resp)

                    #scale baack down by summing:
                    on_parasol_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #convert response to firing rate


    #go through all rgc responses and convert response to firing rate

    #we'll use an exponential function to get spike rate from rgc response (golden got this from pillow et al 2008)
    #large rgc response is 1.1, so exponential goes from e^0=1 to e^1.2=3.00
    #6 Hz seems to be a large neuron firing rate for an RGC (from other studies), so we'll use this as max_rate
    #exponential function: firing_rate = max_rate*((e^(response)-1)/2.00)
    #subtract 1 from exponential so we have values starting from 0 to 2.00, divide by 2.00 and multiply by max rate so
    #large responses will be about equal to our inputted max firing rate (maybe slightly larger)

    max_rate = 6 #6Hz seems to be a very large RGC firing rate

    #on parasol rgcs:

    i=0
    j=0
    while(i<np.shape(on_parasol_responses)[0]):
        while(j<np.shape(on_parasol_responses)[1]):
            response = on_parasol_responses[i,j]

            firing_rate = max_rate*((math.e**(response)-1)/2)

            if(firing_rate < 0):
                firing_rate = 0

            on_parasol_responses[i,j] = firing_rate

            j+=1
        j=0
        i+=1
    
    return on_parasol_responses


#electrode functions

def G_electrodes(r, amp):
    sigma = 35
    G = 1/(2*math.pi*sigma**2) * math.e**(-(r**2)/(2*sigma**2))
    #G = 1 * math.e**(-(r**2)/(2*sigma**2)) # G with amplitude 1
    return G*amp

def simulate_electrodes(electrode_voltages):
    
    #RGC matrices
    on_parasol_responses = np.zeros((64,64))


    #Bipolar matrices
    on_parasol_bipolar_responses = np.zeros((256,256))


    #Electrodes
    #electrode_voltages = np.zeros((8,8))
    
    electrode_help = np.zeros((256,256)) #scaled up electrodes
    
    #place elctrodes on scaled up electrode grid
    i=0
    j=0
    while(i<np.shape(electrode_voltages)[0]):
        while(j<np.shape(electrode_voltages)[1]):
            
            electrode_help[i*32+16, j*32+16] = electrode_voltages[i,j]
            
            j+=1
        j=0
        i+=1
    
    
    # get bipolar responses from electrode outputs

    # scale up by 64 (64 microns each axis = ~32 neurons each way, electrode feeds into surrounding 64x64 bipolars)

    i=0
    j=0
    while(i<np.shape(on_parasol_bipolar_responses)[0]):
        while(j<np.shape(on_parasol_bipolar_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j<16):
                b_init=16-j #only can go over to 0th col of bipolars
            if(i<16):
                a_init=16-i #only can go over to 0th row of bipolars
            if(j>239):
                b_max=16+(255-j) #only can go over to 255th col of bipolars
            if(i>239):
                a_max=16+(255-i) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):
                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = electrode_help[i+(a-16),j+(b-16)]
                    resp = G_electrodes(r, in_resp)

                    #scale back down by summing:
                    on_parasol_bipolar_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #temporal response


    #Since golden paper didn't give any information on the temporal filter (and we are using slightly different
    #temporal modeling with single frame of picture shown then nothing for the rest of the 0.4 s), we will use
    #exponential decay to model the temporal behavior with the initial value starting at the spacial response
    #and the decay value to be 50 (0.02s half life of neuron response strength, close to curve of golden
    #temporal response)

    #response = init_response * e^(decay_val*t)

    decay_val = -50


    #on parasol temporal:

    initial_on_parasol_bipolar_responses = np.copy(on_parasol_bipolar_responses)

    t=0.001 #time steps of 0.001s (400 time periods)
    while t<0.4: #simulate 0.4s like golden paper did
        i=0
        j=0
        while(i<np.shape(on_parasol_bipolar_responses)[0]):
            while(j<np.shape(on_parasol_bipolar_responses)[1]):
                #calculate current time's temporal response value with exponential decay equation, then add it
                #to the overall bipolar response

                init_response = initial_on_parasol_bipolar_responses[i,j]

                response = init_response * math.e**(decay_val*t)

                on_parasol_bipolar_responses[i,j] += response #sum over temporal response for each neuron

                j+=1
            i+=1

        t+=0.001
    
    # get rgc responses from bipolars

    # scale up by 32 from rgc matrix (need 25 microns/axis ~12 neurons each way, means we have 25x25 bipolars feeding each
    # rgc including the bipolar directly on top of the rgc)
    # we need our scaling factor to be a multiple of 4 though (which ignores symmetry), so we will scale by 32x32
    # so that despite assymetry, the value we are losing are essentially 0
    
    i=0
    j=0
    while(i<np.shape(on_parasol_responses)[0]):
        while(j<np.shape(on_parasol_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j*4<16):
                b_init=16-j*4 #only can go over to 0th col of bipolars
            if(i*4<16):
                a_init=16-i*4 #only can go over to 0th row of bipolars
            if(j*4>239):
                b_max=16+(255-j*4) #only can go over to 255th col of bipolars
            if(i*4>239):
                a_max=16+(255-i*4) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):

                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = on_parasol_bipolar_responses[i*4+(a-16),j*4+(b-16)]
                    resp = DoG_on_parasol(r, in_resp)

                    #scale baack down by summing:
                    on_parasol_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #convert response to firing rate


    #go through all rgc responses and convert response to firing rate

    #we'll use an exponential function to get spike rate from rgc response (golden got this from pillow et al 2008)
    #large rgc response is 1.1, so exponential goes from e^0=1 to e^1.2=3.00
    #6 Hz seems to be a large neuron firing rate for an RGC (from other studies), so we'll use this as max_rate
    #exponential function: firing_rate = max_rate*((e^(response)-1)/2.00)
    #subtract 1 from exponential so we have values starting from 0 to 2.00, divide by 2.00 and multiply by max rate so
    #large responses will be about equal to our inputted max firing rate (maybe slightly larger)

    max_rate = 6 #6Hz seems to be a very large RGC firing rate

    #on parasol rgcs:

    i=0
    j=0
    while(i<np.shape(on_parasol_responses)[0]):
        while(j<np.shape(on_parasol_responses)[1]):
            response = on_parasol_responses[i,j]

            firing_rate = max_rate*((math.e**(response)-1)/2)

            if(firing_rate < 0):
                firing_rate = 0

            on_parasol_responses[i,j] = firing_rate

            j+=1
        j=0
        i+=1
    
    return on_parasol_responses




def DoG_off_parasol_bipolars(r, amp):
    a_cent = 1
    a_sur = 0.01
    sig_cent = 4
    sig_sur = sig_cent * 9
    DoG = a_cent * math.e**(-(r**2)/(2*sig_cent**2)) - a_sur * math.e**(-(r**2)/(2*sig_sur**2))
    return DoG*amp

def DoG_off_parasol(r, amp):
    a_cent = 1
    a_sur = 0.375
    sig_cent = 8
    sig_sur = sig_cent * 1.15
    DoG = a_cent * math.e**(-(r**2)/(2*sig_cent**2)) - a_sur * math.e**(-(r**2)/(2*sig_sur**2))
    return DoG*amp

def simulate_off(cone_array):
    
    #RGC matrices
    off_parasol_responses = np.zeros((64,64))


    #Bipolar matrices
    off_parasol_bipolar_responses = np.zeros((256,256))


    #Cone matrices
    cone_responses = np.copy(cone_array)
    
    
    
    # get bipolar responses from cone outputs

    # scale up by 11 (10 microns each axis = 5 neurons each way, this means we have 11x11 cones feeding into each bipolar
    # including the cone directly on top of the bipolar)

    i=0
    j=0
    while(i<np.shape(off_parasol_bipolar_responses)[0]):
        while(j<np.shape(off_parasol_bipolar_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 11
            b_max = 11

            if(j<5):
                b_init=5-j #only can go over to 0th col of cones
            if(i<5):
                a_init=5-i #only can go over to 0th row of cones
            if(j>250):
                b_max=6+(255-j) #only can go over to 255th col of cones
            if(i>250):
                a_max=6+(255-i) #only can go over to 255th row of cones

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):
                    #scale up and get radii:
                    r = math.sqrt(((a-5)**2) + ((b-5)**2))

                    #get scaled up response:
                    in_resp = cone_responses[i+(a-5),j+(b-5)]
                    resp = DoG_off_parasol_bipolars(r, in_resp)

                    #scale back down by summing:
                    off_parasol_bipolar_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    #temporal response


    #Since golden paper didn't give any information on the temporal filter (and we are using slightly different
    #temporal modeling with single frame of picture shown then nothing for the rest of the 0.4 s), we will use
    #exponential decay to model the temporal behavior with the initial value starting at the spacial response
    #and the decay value to be -50 (0.02s half life of neuron response strength, close to curve of golden
    #temporal response)

    #response = init_response * e^(decay_val*t)

    decay_val = -50


    #off parasol temporal:

    initial_off_parasol_bipolar_responses = np.copy(off_parasol_bipolar_responses)

    t=0.001 #time steps of 0.001s (400 time periods)
    while t<0.4: #simulate 0.4s like golden paper did
        i=0
        j=0
        while(i<np.shape(off_parasol_bipolar_responses)[0]):
            while(j<np.shape(off_parasol_bipolar_responses)[1]):
                #calculate current time's temporal response value with exponential decay equation, then add it
                #to the overall bipolar response

                init_response = initial_off_parasol_bipolar_responses[i,j]

                response = init_response * math.e**(decay_val*t)

                off_parasol_bipolar_responses[i,j] += response #sum over temporal response for each neuron

                j+=1
            i+=1

        t+=0.001
    
    # get rgc responses from bipolars

    # scale up by 32 from rgc matrix (need 25 microns/axis ~12 neurons each way, means we have 25x25 bipolars feeding each
    # rgc including the bipolar directly on top of the rgc)
    # we need our scaling factor to be a multiple of 4 though (which ignores symmetry), so we will scale by 32x32
    # so that despite assymetry, the value we are losing are essentially 0
    
    i=0
    j=0
    while(i<np.shape(off_parasol_responses)[0]):
        while(j<np.shape(off_parasol_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j*4<16):
                b_init=16-j*4 #only can go over to 0th col of bipolars
            if(i*4<16):
                a_init=16-i*4 #only can go over to 0th row of bipolars
            if(j*4>239):
                b_max=16+(255-j*4) #only can go over to 255th col of bipolars
            if(i*4>239):
                a_max=16+(255-i*4) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):

                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = off_parasol_bipolar_responses[i*4+(a-16),j*4+(b-16)]
                    resp = DoG_off_parasol(r, in_resp)

                    #scale baack down by summing:
                    off_parasol_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #convert response to firing rate


    #go through all rgc responses and convert response to firing rate

    #we'll use an exponential function to get spike rate from rgc response (golden got this from pillow et al 2008)
    #large rgc response is -1.1, so exponential goes from e^0=1 to e^1.2)=3.00
    #6 Hz seems to be a large neuron firing rate for an RGC (from other studies), so we'll use this as max_rate
    #exponential function: firing_rate = max_rate*((e^(-response)-1)/2.00)
    #subtract 1 from exponential so we have values starting from 0 to 2.00, divide by 2.00 and multiply by max rate so
    #large responses will be about equal to our inputted max firing rate (maybe slightly larger)

    max_rate = 6 #6Hz seems to be a very large RGC firing rate

    #off parasol rgcs:

    i=0
    j=0
    while(i<np.shape(off_parasol_responses)[0]):
        while(j<np.shape(off_parasol_responses)[1]):
            response = off_parasol_responses[i,j]

            firing_rate = max_rate*((math.e**(response)-1)/2)

            if(firing_rate < 0):
                firing_rate = 0

            off_parasol_responses[i,j] = firing_rate

            j+=1
        j=0
        i+=1
    
    return off_parasol_responses


#electrode functions

def simulate_electrodes_off(electrode_voltages):
    
    #RGC matrices
    off_parasol_responses = np.zeros((64,64))


    #Bipolar matrices
    off_parasol_bipolar_responses = np.zeros((256,256))


    #Electrodes
    #electrode_voltages = np.zeros((8,8))
    
    electrode_help = np.zeros((256,256)) #scaled up electrodes
    
    #place elctrodes on scaled up electrode grid
    i=0
    j=0
    while(i<np.shape(electrode_voltages)[0]):
        while(j<np.shape(electrode_voltages)[1]):
            
            electrode_help[i*32+16, j*32+16] = electrode_voltages[i,j]
            
            j+=1
        j=0
        i+=1
    
    
    # get bipolar responses from electrode outputs

    # scale up by 64 (64 microns each axis = ~32 neurons each way, electrode feeds into surrounding 64x64 bipolars)

    i=0
    j=0
    while(i<np.shape(off_parasol_bipolar_responses)[0]):
        while(j<np.shape(off_parasol_bipolar_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j<16):
                b_init=16-j #only can go over to 0th col of bipolars
            if(i<16):
                a_init=16-i #only can go over to 0th row of bipolars
            if(j>239):
                b_max=16+(255-j) #only can go over to 255th col of bipolars
            if(i>239):
                a_max=16+(255-i) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):
                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = electrode_help[i+(a-16),j+(b-16)]
                    resp = G_electrodes(r, in_resp)

                    #scale back down by summing:
                    off_parasol_bipolar_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #temporal response


    #Since golden paper didn't give any information on the temporal filter (and we are using slightly different
    #temporal modeling with single frame of picture shown then nothing for the rest of the 0.4 s), we will use
    #exponential decay to model the temporal behavior with the initial value starting at the spacial response
    #and the decay value to be 50 (0.02s half life of neuron response strength, close to curve of golden
    #temporal response)

    #response = init_response * e^(decay_val*t)

    decay_val = -50


    #on parasol temporal:

    initial_off_parasol_bipolar_responses = np.copy(off_parasol_bipolar_responses)

    t=0.001 #time steps of 0.001s (400 time periods)
    while t<0.4: #simulate 0.4s like golden paper did
        i=0
        j=0
        while(i<np.shape(off_parasol_bipolar_responses)[0]):
            while(j<np.shape(off_parasol_bipolar_responses)[1]):
                #calculate current time's temporal response value with exponential decay equation, then add it
                #to the overall bipolar response

                init_response = initial_off_parasol_bipolar_responses[i,j]

                response = init_response * math.e**(decay_val*t)

                off_parasol_bipolar_responses[i,j] += response #sum over temporal response for each neuron

                j+=1
            i+=1

        t+=0.001
    
    # get rgc responses from bipolars

    # scale up by 32 from rgc matrix (need 25 microns/axis ~12 neurons each way, means we have 25x25 bipolars feeding each
    # rgc including the bipolar directly on top of the rgc)
    # we need our scaling factor to be a multiple of 4 though (which ignores symmetry), so we will scale by 32x32
    # so that despite assymetry, the value we are losing are essentially 0
    
    i=0
    j=0
    while(i<np.shape(off_parasol_responses)[0]):
        while(j<np.shape(off_parasol_responses)[1]):

            a_init = 0
            b_init = 0
            a_max = 32
            b_max = 32

            if(j*4<16):
                b_init=16-j*4 #only can go over to 0th col of bipolars
            if(i*4<16):
                a_init=16-i*4 #only can go over to 0th row of bipolars
            if(j*4>239):
                b_max=16+(255-j*4) #only can go over to 255th col of bipolars
            if(i*4>239):
                a_max=16+(255-i*4) #only can go over to 255th row of bipolars

            a=a_init
            b=b_init
            while(a < a_max):
                while(b < b_max):

                    #scale up and get radii:
                    r = math.sqrt(((a-16)**2) + ((b-16)**2))

                    #get scaled up response:
                    in_resp = off_parasol_bipolar_responses[i*4+(a-16),j*4+(b-16)]
                    resp = DoG_off_parasol(r, in_resp)

                    #scale baack down by summing:
                    off_parasol_responses[i,j] += resp

                    b+=1
                b=b_init
                a+=1

            j+=1
        j=0
        i+=1
    
    
    #convert response to firing rate


    #go through all rgc responses and convert response to firing rate

    #we'll use an exponential function to get spike rate from rgc response (golden got this from pillow et al 2008)
    #large rgc response is 1.1, so exponential goes from e^0=1 to e^1.2=3.00
    #6 Hz seems to be a large neuron firing rate for an RGC (from other studies), so we'll use this as max_rate
    #exponential function: firing_rate = max_rate*((e^(response)-1)/2.00)
    #subtract 1 from exponential so we have values starting from 0 to 2.00, divide by 2.00 and multiply by max rate so
    #large responses will be about equal to our inputted max firing rate (maybe slightly larger)

    max_rate = 6 #6Hz seems to be a very large RGC firing rate

    #off parasol rgcs:

    i=0
    j=0
    while(i<np.shape(off_parasol_responses)[0]):
        while(j<np.shape(off_parasol_responses)[1]):
            response = off_parasol_responses[i,j]

            firing_rate = max_rate*((math.e**(response)-1)/2)

            if(firing_rate < 0):
                firing_rate = 0

            off_parasol_responses[i,j] = firing_rate

            j+=1
        j=0
        i+=1
    
    return off_parasol_responses


def simulate_degenerated(cone_array):
    on_parasol_responses = simulate(cone_array)

    # Random dropout of 30% of RGCs
    
    tmp = on_parasol_responses.reshape(4096)
    
    zeros = np.random.choice(np.arange(4096), replace=False, size=int(4096 * 0.3))
    tmp[zeros] = 0
    
    on_parasol_responses = tmp.reshape((64,64))
    
    return on_parasol_responses


def simulate_electrodes_degenerated(electrode_voltages):
    on_parasol_responses = simulate_electrodes(electrode_voltages)

    # Random dropout of 30% of RGCs
    
    tmp = on_parasol_responses.reshape(4096)
    
    zeros = np.random.choice(np.arange(4096), replace=False, size=int(4096 * 0.3))
    tmp[zeros] = 0
    
    on_parasol_responses = tmp.reshape((64,64))
    
    return on_parasol_responses


def simulate_degenerated_off(cone_array):
    off_parasol_responses = simulate_off(cone_array)

    # Random dropout of 30% of RGCs
    
    tmp = off_parasol_responses.reshape(4096)
    
    zeros = np.random.choice(np.arange(4096), replace=False, size=int(4096 * 0.3))
    tmp[zeros] = 0
    
    off_parasol_responses = tmp.reshape((64,64))
    
    return off_parasol_responses


def simulate_electrodes_degenerated_off(electrode_voltages):
    off_parasol_responses = simulate_electrodes_off(electrode_voltages)

    # Random dropout of 30% of RGCs
    
    tmp = off_parasol_responses.reshape(4096)
    
    zeros = np.random.choice(np.arange(4096), replace=False, size=int(4096 * 0.3))
    tmp[zeros] = 0
    
    off_parasol_responses = tmp.reshape((64,64))
    
    return off_parasol_responses


# Grabbing Images from Pickle File (file is > 1gb, not included on github) ------------------------------------------------------------------

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

imgs = unpickle('D:/ImageNet/Imagenet64_train_part1/train_data_batch_1')
imgs['data'].shape



# Simulation and Training --------------------------------------------------------------------------------------------------------------------


#on pathway

test_imgs_tmp_2000 = np.copy(imgs['data'][0:2000])
test_imgs_2000 = np.zeros((2000,256*256))
retina_responses_2000 = np.zeros((2000,64*64))

i=0
for row in test_imgs_tmp_2000:
    image = row.reshape((64,64,3), order='F').transpose(1,0,2)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (256,256), interpolation = cv2.INTER_AREA)
    test_imgs_2000[i] = resized.reshape(256*256)
    
    cone_array = resized * (0.0001/255.0) #grayscale 0 to 0.0001, cones set up to have responses up to 0.0001
    response = simulate(cone_array)
    retina_responses_2000[i] = response.reshape(64*64)
    i+=1


ridge_2000 = Ridge().fit(retina_responses_2000[0:1999], test_imgs_2000[0:1999])


#off pathway

off_imgs_2000 = np.zeros((2000,256*256))
off_responses_2000 = np.zeros((2000,64*64))

i=0
for row in test_imgs_2000:
    off_imgs_2000[i] = (row*-1 + 255)
    
    cone_array = off_imgs_2000 * (0.0001/255.0) #grayscale 0 to 0.0001, cones set up to have responses up to 0.0001
    cone_array = cone_array.reshape((256,256))
    off_response = simulate_off(cone_array)
    off_responses_2000[i] = off_response.reshape(64*64)
    i+=1


ridge_off_2000 = Ridge().fit(off_responses_2000[0:1999], off_imgs_2000[0:1999])



# Running images through the network and comparing reconstructions/saving images ----------------------------------------------------------


#glasses natural vision

test_output_ridge = ridge_2000.predict(retina_responses_2000[1999].reshape((1,64*64)))
test_output_ridge_off = ridge_off_2000.predict(off_responses_2000[1999].reshape((1,64*64)))

plt.imshow(retina_responses_2000[1999].reshape((64,64)), cmap='gray')
plt.title('natural vision on rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_natural_vision_on_activity.png')

plt.imshow(off_responses_2000[1999].reshape((64,64)), cmap='gray')
plt.title('natural vision off rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_natural_vision_off_activity.png')

plt.imshow(test_output_ridge[0].reshape((256,256)), cmap='gray')
plt.title('natural vision on reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_natural_vision_on.png')

plt.imshow(test_output_ridge_off[0].reshape((256,256)), cmap='gray')
plt.title('natural vision off reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_natural_vision_off.png')

plt.imshow((test_output_ridge[0]-test_output_ridge_off[0]).reshape((256,256)), cmap='gray')
plt.title('natural vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_natural_vision_full.png')

plt.imshow(test_imgs_2000[1999].reshape((256,256)), cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_original.png')


#glasses electrodes

voltages = cv2.resize(test_imgs_2000[1999].reshape(256,256), (8,8), interpolation = cv2.INTER_AREA)

voltages = voltages * (50/255.0) #grayscale 0 to 50, electrode set up to have voltages up to 50V
on_elec_responses = simulate_electrodes(voltages)
off_elec_responses = simulate_electrodes_off(voltages)

on_elec_reconstruction = ridge_2000.predict(on_elec_responses.reshape((1,64*64)))
off_elec_reconstruction = ridge_off_2000.predict(off_elec_responses.reshape((1,64*64)))



plt.imshow(on_elec_responses, cmap='gray')
plt.title('prosthetic vision on rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_prosthetic_vision_on_activity.png')

plt.imshow(off_elec_responses, cmap='gray')
plt.title('prosthetic vision off rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_prosthetic_vision_off_activity.png')

plt.imshow(on_elec_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('prosthetic vision on reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_prosthetic_vision_on.png')

plt.imshow(off_elec_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('prosthetic vision off reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_prosthetic_vision_off.png')

plt.imshow((on_elec_reconstruction[0]-off_elec_reconstruction[0]).reshape((256,256)), cmap='gray')
plt.title('prosthetic vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_prosthetic_vision_full.png')

plt.imshow(test_imgs_2000[1999].reshape((256,256)), cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_original.png')



#glasses degenerated natural vision

on_cone_array = test_imgs_2000[1999].reshape((256,256)) * (0.0001/255.0)
off_cone_array = off_imgs_2000[1999].reshape((256,256)) * (0.0001/255.0)

on_responses = simulate_degenerated(on_cone_array)
off_responses = simulate_degenerated_off(off_cone_array)

on_reconstruction = ridge_2000.predict(on_responses.reshape((1,64*64)))
off_reconstruction = ridge_off_2000.predict(off_responses.reshape((1,64*64)))



plt.imshow(on_responses, cmap='gray')
plt.title('degenerated natural vision on rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degenerated_vision_on_activity.png')

plt.imshow(off_responses, cmap='gray')
plt.title('degenerated natural vision off rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degenerated_vision_off_activity.png')

plt.imshow(on_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('degenerated natural vision on reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degenerated_vision_on.png')

plt.imshow(off_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('degenerated natural vision off reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degenerated_vision_off.png')

plt.imshow((on_reconstruction[0]-off_reconstruction[0]).reshape((256,256)), cmap='gray')
plt.title('degenerated natural vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degenerated_vision_full.png')

plt.imshow(test_imgs_2000[1999].reshape((256,256)), cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_original.png')



#glasses degenerated electrode vision

voltages = cv2.resize(test_imgs_2000[1999].reshape(256,256), (8,8), interpolation = cv2.INTER_AREA)

voltages = voltages * (50/255.0) #grayscale 0 to 50, electrode set up to have voltages up to 50V
on_elec_responses = simulate_electrodes_degenerated(voltages)
off_elec_responses = simulate_electrodes_degenerated_off(voltages)

on_elec_reconstruction = ridge_2000.predict(on_elec_responses.reshape((1,64*64)))
off_elec_reconstruction = ridge_off_2000.predict(off_elec_responses.reshape((1,64*64)))



plt.imshow(on_elec_responses, cmap='gray')
plt.title('degenerated prosthetic vision on rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degerated_prosthetic_vision_on_activity.png')

plt.imshow(off_elec_responses, cmap='gray')
plt.title('degenerated prosthetic vision off rgc activity')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degerated_prosthetic_vision_off_activity.png')

plt.imshow(on_elec_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('degenerated prosthetic vision on reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degerated_prosthetic_vision_on.png')

plt.imshow(off_elec_reconstruction[0].reshape((256,256)), cmap='gray')
plt.title('degenerated prosthetic vision off reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degerated_prosthetic_vision_off.png')

plt.imshow((on_elec_reconstruction[0]-off_elec_reconstruction[0]).reshape((256,256)), cmap='gray')
plt.title('degenerated prosthetic vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_degerated_prosthetic_vision_full.png')

plt.imshow(test_imgs_2000[1999].reshape((256,256)), cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/glasses_original.png')


#sundial natural vision

image = imgs['data'][2001].reshape((64,64,3), order='F').transpose(1,0,2)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray_image, (256,256), interpolation = cv2.INTER_AREA)

on_cone_array = resized * (0.0001/255.0) #grayscale 0 to 0.0001, cones set up to have responses up to 0.0001
off_cone_array = (resized*-1 + 255) * (0.0001/255.0)

on_responses = simulate(on_cone_array)
off_responses = simulate_off(off_cone_array)

on_reconstruction = ridge_2000.predict(on_responses.reshape((1,64*64)))
off_reconstruction = ridge_off_2000.predict(off_responses.reshape((1,64*64)))


plt.imshow((on_reconstruction[0]-off_reconstruction[0]).reshape((256,256)), cmap='gray')
plt.title('natural vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/sundial_natural_vision_full.png')
#plt.show()

plt.imshow(resized, cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/sundial_original.png')
#plt.show()


#sundial electrodes

image = imgs['data'][2001].reshape((64,64,3), order='F').transpose(1,0,2)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray_image, (256,256), interpolation = cv2.INTER_AREA)

voltages = cv2.resize(resized, (8,8), interpolation = cv2.INTER_AREA)

voltages = voltages * (50/255.0) #grayscale 0 to 50, electrode set up to have voltages up to 50V
on_elec_responses = simulate_electrodes(voltages)
off_elec_responses = simulate_electrodes_off(voltages)

on_elec_reconstruction = ridge_2000.predict(on_elec_responses.reshape((1,64*64)))
off_elec_reconstruction = ridge_off_2000.predict(off_elec_responses.reshape((1,64*64)))



plt.imshow((on_elec_reconstruction[0]-off_elec_reconstruction[0]).reshape((256,256)), cmap='gray')
plt.title('prosthetic vision full (on-off) reconstruction')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/sundial_prosthetic_vision_full.png')
#plt.show()

plt.imshow(resized.reshape((256,256)), cmap='gray')
plt.title('original image')
plt.savefig('C:/Users/rhyst/Documents/CS291A/samples/sundial_original.png')
#plt.show()

