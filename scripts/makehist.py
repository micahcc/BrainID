#!/usr/bin/python

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
def HRF(stims, TR, num):
    TR = float(TR)
    inputl = 0;
    inputt = 0
    dt = TR/HRFDIVIDER
    stimarr = [i for i in range(0, int(num*HRFDIVIDER + 2*hrfparam[6]/dt))]
    for i in range(0, len(stimarr)):
        if inputt < len(stims) and i*dt > stims[inputt][0] + hrfparam[5]\
                    + hrfparam[6]:
            inputl = stims[inputt][1]
            inputt = inputt+1
        stimarr[i] = inputl

    points = [i for i in range(0,hrfparam[6]/dt)]
    hrf = [gamma.pdf(u, hrfparam[0]/hrfparam[2], scale=hrfparam[2]/dt) - \
                gamma.pdf(u, hrfparam[1]/hrfparam[3], scale=hrfparam[3]/dt) \
                / hrfparam[4] for u in points]
    scale = sum([hrf[i] for i in range(len(hrf)) if i%HRFDIVIDER == 0])
    hrf = [point/scale for point in hrf]
    
    hdsignal = convolve(stimarr, hrf, mode="full") 
    signal = [hdsignal[i] for i in range(0, len(stimarr)) \
                if i%HRFDIVIDER == 0 and i*dt > hrfparam[6] and \
                i*dt < len(stimarr)*dt-hrfparam[6]]
    delta = [(hdsignal[i+1] - hdsignal[i])/dt for i in range(0, len(stimarr)) \
                if i%HRFDIVIDER == 0 and i*dt > hrfparam[6] and \
                i*dt < len(stimarr)*dt-hrfparam[6]]
    return (signal, delta)

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
    PARAM = -1
    SHIFT=0
    param = PARAM
    if len(sys.argv) == 6:
        try:
            pos = [float(arg) for arg in sys.argv[2:5]]
            param = float(sys.argv[5])
        except:
            param = PARAM 
            pos = (0,0,0);
    elif len(sys.argv) == 2:
        param = PARAM 
        pos = (0,0,0);
    else: 
        print "Usage: ", sys.argv[0], "<InDir> [position + param]"
        print "Looks in Dir for: "
        print "stim0, stim1 (must be shifted to match pfilter_input), histogram.nii.gz"
        print "pfilter_input.nii.gz, truebold.nii.gz, truestate.nii.gz"
        sys.exit(-1);
    
    actual = nibabel.load(sys.argv[1] + "pfilter_input.nii.gz")
    histimg = nibabel.load(sys.argv[1] + "histogram.nii.gz")
    #beta1 = nibabel.load(sys.argv[1] + "beta_0001.img")
    #beta2 = nibabel.load(sys.argv[1] + "beta_0002.img")
    #beta3 = nibabel.load(sys.argv[1] + "beta_0003.img")
    #beta4 = nibabel.load(sys.argv[1] + "beta_0004.img")
    #beta5 = nibabel.load(sys.argv[1] + "beta_0005.img")
    
    TR = histimg.get_header()['pixdim'][4];
    if TR == 1:
        print "Pixdim is 1, changing to 2.1"
        TR = 2.1
    
    #SPM = io.loadmat(sys.argv[1]+"SPM.mat")
    #design = SPM['SPM'][0,0].xX[0,0].X
    #b1 = beta1.get_data()[35,14,7]
    #b2 = beta2.get_data()[35,14,7]
    #b3 = beta3.get_data()[35,14,7]
    #b4 = beta4.get_data()[35,14,7]
    #b5 = beta5.get_data()[35,14,7]
    #
    #d1 = [b1*samp for samp in design[:,0]]
    #d2 = [b2*samp for samp in design[:,1]]
    #d3 = [b3*samp for samp in design[:,2]]
    #d4 = [b4*samp for samp in design[:,3]]
    #d5 = [b5*samp for samp in design[:,4]]
    #
    #result = map(lambda a,b,c,d : a+b+c+d, d1, d2, d3, d4)
    #
    #P.plot(result)
    #P.legend(["actual", "sum"])
    #P.show()
    #sys.exit()
    
    stims = readStim(sys.argv[1]+"stim")
    histograms = processHistos(histimg, pos, param)
    plothisto(histograms, TR)
    
    if param < 0:
        P.plot([i*TR for i in range(actual.get_header()['dim'][4])], \
                    actual.get_data()[pos[0],pos[1],pos[2],:], '-*');
        leg = ["actual"]
        try:
            truebold = nibabel.load(sys.argv[1] + "truebold.nii.gz")
            print "Ground truth for bold found"
            P.plot([i*TR for i in range(truebold.get_header()['dim'][4])], \
                        truebold.get_data()[pos[0],pos[1],pos[2],:], 'g-+');
            leg.append("truth")
        except:
            print "No ground truth for bold available, if you have some put it in ", \
                        sys.argv[1] + "truebold.nii.gz"
        print "params"
        print getparams_mu(-1, histimg, pos)
        print getA(getparams_mu(-1, histimg, pos)[2])
        exp_final= sim(stims, getparams_mu(-1, histimg, pos), TR, actual.get_header()['dim'][4])
        print len(exp_final)
        
        P.plot([i*TR for i in range(len(exp_final))], exp_final)
        times = [val[0] for val in stims]
        times2 = [val[0] - .00001 for val in stims]
        times.extend(times2);
        times = sorted(times)
        del(times[0])
        levels = [(i%4)/2 for i in range(0, len(times))]
        stims = zip(times,levels)

        P.plot([val[0] for val in stims], [-.01+.005*val[1] for val in stims], 'r')
        leg.extend(["FinalMu", "Stimuli"])
        P.legend(leg)
        
    else:
        try:
            truestate = nibabel.load(sys.argv[1] + "truestate.nii.gz")
            P.plot([i*TR for i in range(truestate.get_header()['dim'][4])], \
                        truestate.get_data()[0,param,0,:], 'g-+');
            print "Truth", truestate.get_data()[pos[0],pos[1],pos[2],param];
            leg = ["truth"]
            P.legend(leg)
        except:
            print "No ground truth for bold available, if you have some put it in ", \
                        sys.argv[1] + "truestate.nii.gz"
    
    #P.plot(initest)
    #P.plot(measmu.get_data()[0,0,0,:]);
    #P.plot(final)
    #P.plot(canonical)
    
    
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


