#!/usr/bin/python

import matplotlib
import pylab as P
import sys
import nibabel 
from paramtest import State, readout, transition, getA
from numpy import convolve
import scipy.stats
from scipy.stats.distributions import gamma
import scipy.io as io
from bar import histo, plothisto
from math import isinf

DIVIDER = 4096
HRFDIVIDER = 16

NAMES = ['$\\tau_0$', '$\\alpha$', '$E_0$', '$V_0$', '$\\tau_s$', '$\\tau_f$', \
            '$\epsilon$', '$V_T$', '$Q_T$', '$S_T$', '$F_T$', 'BOLD\\%']

matplotlib.rc('text', usetex=True)

#t = arange(0.0, 2.0, 0.01)
#s = sin(2*pi*t)
#plot(t, s, linewidth=1.0)
#
#xlabel('time (s)')
#ylabel('voltage (mV)')
#title('About as simple as it gets, folks')
#grid(True)
#show()

def getparams_mode(time, source, pos):
    params = [0 for i in range(0,7)]
    for pp in range(0, 7):
        upper = source.get_data()[pos[0],pos[1],pos[2],time,pp, -2]
        lower = source.get_data()[pos[0],pos[1],pos[2],time,pp, -3]
        count = source.get_header()['dim'][6] - 3
        width = (upper-lower)/count
        print lower, upper, width, count
        histinput = [[] for j in range(0, histimg.get_header()['dim'][6]-3)]
        for j in range(0, histimg.get_header()['dim'][6]-3):
            histinput[j] = [histimg.get_data()[pos[0],pos[1],pos[2],time, pp, j], lower+width/2.+j*width]
        params[pp] = max(histinput, key=lambda pair: pair[0])[1]
    return params

def getparams_mu(time, source, pos):
    return source.get_data()[pos[0],pos[1],pos[2], time, 0:7, -1] 

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
        if inputt < len(stims) and t*TR/DIVIDER+SHIFT*TR > stims[inputt][0]:
            inputl = stims[inputt][1]
            inputt = inputt+1
        statevars = transition(statevars, params, TR/DIVIDER, inputl);
        if t%DIVIDER == 0:
            out[t/DIVIDER] = readout(statevars, params)
    return out

hrfparam   = [6, 16, 1, 1, 6, 0, 32];

def readStim(filename):
    stimin = open(filename)
    stims = [[float(var) for var in line.split()] for line in stimin.readlines()]
    print stims
    return stims

def processHistos(image, pos, p):
    out = [[] for t in range(0, image.get_header()['dim'][4])]
    #generate the histogram for measurements
    for t in range(0, image.get_header()['dim'][4]):
        upper = image.get_data()[pos[0],pos[1],pos[2],t,p, -2]
        lower = image.get_data()[pos[0],pos[1],pos[2],t,p, -3]
        count = image.get_header()['dim'][6] - 3
        width = (upper-lower)/count
    #    print lower, upper, count, width
        if isinf(lower) :
            print image.get_data()[pos[0],pos[1],pos[2],t, p, 0:-3]
        histinput = [[] for j in range(0, image.get_header()['dim'][6]-3)]
        for j in range(0, image.get_header()['dim'][6]-3):
            histinput[j] = [image.get_data()[pos[0],pos[1],pos[2],t, p, j], lower+width*j, \
                        lower+width*(j+1)]
        out[t] = histo(histinput)

    return out 

if __name__ == "__main__":
    #Begin Main, main
    if len(sys.argv) != 2:
        print "Usage: %s <directory>" % sys.argv[0];
        print "Looks in Dir for: "
        print "stim0, stim1 (must be shifted to match pfilter_input), histogram.nii.gz"
        print "pfilter_input.nii.gz, truebold.nii.gz, truestate.nii.gz"
        sys.exit(-1);
    
#    actual = nibabel.load(sys.argv[1] + "pfilter_input.nii.gz")
    histimg = nibabel.load(sys.argv[1] + "histogram.nii.gz")
    truebold = nibabel.load(sys.argv[1] + "truebold.nii.gz")
    truestate = nibabel.load(sys.argv[1] + "truestate.nii.gz")
    stims = readStim(sys.argv[1]+"stim");
    
    TR = histimg.get_header()['pixdim'][4];
    if TR == 1:
        print "Pixdim is 1, changing to 2.1"
        TR = 2.1
    
    pos = [0,0,0]
    ploc = 0
    for param in range(0, 12):
        if param%4 == 0:
            P.show()

        P.subplot(4,1, param%4+1)
        print NAMES[param]
        P.title(NAMES[param])
        if param != 11:
            P.plot([i*TR for i in range(truestate.get_header()['dim'][4])], 
                        truestate.get_data()[0, param, 0, :])
            if param%4 == 0:
                P.legend(["True"])
            histograms = processHistos(histimg, pos, param)
            plothisto(histograms, TR)
        else:
            P.plot([i*TR for i in range(truebold.get_header()['dim'][4])], 
                        truebold.get_data()[0, 0, 0, :])
            if param%4 == 0:
                P.legend(["True"])
            histograms = processHistos(histimg, pos, -1)
            plothisto(histograms, TR)

    P.show()
    #P.plot(initest)
    #P.plot(measmu.get_data()[0,0,0,:]);
    #P.plot(final)
    #P.plot(canonical)
    
    
    #
    #names = ['Ts', 'Tf', 'epsilon', 'T0', 'alpha', 'E_0', 'V0', 'Vt', 'Qt', 'St', 'Ft']
    #for i in range(0,11):
    #    P.subplot(6, 2, i+1)
    #    line1 = P.plot(truestat.data[:, 0, i, 0])
    #    line2 = P.plot(eststat.data[:, 0, i, 0])
    #    P.ylabel(names[i])
    #
    #P.show()


