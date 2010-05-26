#!/usr/bin/python

import pylab as P
import sys
import nibabel 
from paramtest import State, readout, transition
from numpy import convolve
import scipy.stats
from scipy.stats.distributions import gamma

DIVIDER = 512

#t = arange(0.0, 2.0, 0.01)
#s = sin(2*pi*t)
#plot(t, s, linewidth=1.0)
#
#xlabel('time (s)')
#ylabel('voltage (mV)')
#title('About as simple as it gets, folks')
#grid(True)
#show()

def printparams(params):
    print "TAU_0  " , params[0]
    print "ALPHA  " , params[1]
    print "E_0    " , params[2]
    print "V_0    " , params[3]
    print "TAU_S  " , params[4]
    print "TAU_F  " , params[5]
    print "EPSILON" , params[6]
    print "A_1    " , params[7]
    print "A_2    " , params[8]



def sim(stims, params, TR, num):
    TR = float(TR)
    out = [0 for i in range(0, num)]
    inputl = 0;
    inputt = 0
    statevars = State()
    stopt = num*TR/DIVIDER
    for t in range(0, num*DIVIDER):
        if inputt < len(stims) and t*TR/DIVIDER > stims[inputt][0]:
            inputl = stims[inputt][1]
            inputt = inputt+1
        statevars = transition(statevars, params, TR/DIVIDER, inputl);
        if t%DIVIDER == 0:
            out[t/DIVIDER] = readout(statevars, params)
    return out

hrfparam   = [6, 16, 1, 1, 6, 0, 32];
def HRF(stims, TR, num):
    TR = float(TR)
    inputl = 0;
    inputt = 0
    dt = TR/DIVIDER
    stimarr = [i for i in range(0, int(num*DIVIDER + 2*hrfparam[6]/dt))]
    for i in range(0, len(stimarr)):
        if inputt < len(stims) and i*dt > stims[inputt][0] + hrfparam[5]\
                    + hrfparam[6]:
            inputl = stims[inputt][1]
            inputt = inputt+1
        stimarr[i] = inputl

    points = [i for i in range(0,hrfparam[6]/dt)]
    hrf = [gamma.pdf(u, hrfparam[0]/hrfparam[2], scale=hrfparam[2]/dt) - \
                gamma.pdf(u, hrfparam[1]/hrfparam[3], scale=hrfparam[3]/dt)/hrfparam[4] \
                for u in points]
    
    hdsignal = convolve(stimarr, hrf, mode="full") 
    
    return [hdsignal[i] for i in range(0, len(stimarr)) \
                if i%DIVIDER == 0 and i*dt > hrfparam[6] and i*dt < len(stimarr)*dt-hrfparam[6]]

    

if len(sys.argv) != 2:
    print "Usage: ", sys.argv[0], "<InDir>"
    print "Looks in Dir for: "
    print "stim (must be shifted to match pfilter_input), meas_mu.nii.gz, meas_var.nii.gz "
    print "param_mu.nii.gz, param_var.nii.gz and pfilter_input.nii.gz"
    sys.exit(-1);

measmu = nibabel.load(sys.argv[1] + "meas_mu.nii.gz")
measvar = nibabel.load(sys.argv[1] + "meas_var.nii.gz")
parammu = nibabel.load(sys.argv[1] + "param_mu.nii.gz")
paramvar = nibabel.load(sys.argv[1] + "param_var.nii.gz")
real = nibabel.load(sys.argv[1] + "pfilter_input.nii.gz")

#print "measmu", measmu
#print "real", real
#print "parammu", parammu

stimfile = open(sys.argv[1] + "stim")
stims = [(float(line.split()[0]), float(line.split()[1])) for line in stimfile.readlines()]
#print stims

#print real
TR = measmu.get_header()['pixdim'][4];
if TR == 1:
    print "Pixdim is 1, changing to 2.1"
    TR = 2.1
print "Initial Params"
printparams(parammu.get_data()[0,0,0,0,:])
print "Final Params"
printparams(parammu.get_data()[0,0,0,-1,:])
canonical = HRF(stims, TR, measmu.get_header()['dim'][4])
initest = sim(stims, parammu.get_data()[0,0,0,0,:], TR, measmu.get_header()['dim'][4])
final = sim(stims, parammu.get_data()[0,0,0,-1,:], TR, measmu.get_header()['dim'][4])


adjusted = [real.get_data()[0,0,0,ii] + parammu.get_data()[0,0,0,-1, 11] \
            for ii in range(0, len(real.get_data()[0,0,0,:]))]
    
P.plot(adjusted, '-*');
P.plot(initest)
P.plot(measmu.get_data()[0,0,0,:]);
P.plot(final)
P.plot(canonical)

P.legend(["Adjusted real", "Estimated", "Final Est", "HRF"])

P.show()
#
#names = ['Ts', 'Tf', 'epsilon', 'T0', 'alpha', 'E_0', 'V0', 'Vt', 'Qt', 'St', 'Ft']
#for i in range(0,11):
#    P.subplot(6, 2, i+1)
#    line1 = P.plot(truestat.data[:, 0, i, 0])
#    line2 = P.plot(eststat.data[:, 0, i, 0])
#    P.ylabel(names[i])
#
#P.show()

