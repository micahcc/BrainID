import pylab as P
import os
from nifti import *

def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
    while True:
        ok = raw_input(prompt)
        if ok in ('y', 'ye', 'yes'): return True
        if ok in ('n', 'no', 'nop', 'nope'): return False
        retries = retries - 1
        if retries < 0: raise IOError, 'refusenik user'
        print complaint



STIM = 'stim.in'
SAMPLERATE = 2
SIMRATE = .001
#ENDTIME = 1800

PARALLEL = 6

truebold = image.NiftiImage('simseries')
truestat = image.NiftiImage('simstate')

estbold = image.NiftiImage('bold')
eststat = image.NiftiImage('state')

#kludge
truebold.setRepetitionTime(2)

yesno = 1
if os.path.exists("mse.nii.gz"):
    yesno = ask_ok("mse.nii.gz exists, regenerate? ", 300)

if yesno == True:
    init = list(truestat.data[0, 0, 7:11, 0])
    forks = 1
    os.system('mkdir data')
    for i in range(0,eststat.timepoints):
        params = list(eststat.data[i, 0, 0:7, 0]) + init
    #    mylist = ['../boldgen', '-i', STIM, '-t', str(SAMPLERATE), '-s', str(SIMRATE), '-h']
        mylist = ['../boldgen', '-i', STIM, '-t', str(SAMPLERATE), '-s', \
                        str(SIMRATE), '-e', \
                        str((eststat.timepoints-1)*truebold.getRepetitionTime()), \
                        '-o', "data/resim%04d.nii.gz" % i, '-p', \
                        "%f %f %f %f %f %f %f %f %f %f %f" % tuple(params)]
        print mylist
        
        if forks >= PARALLEL:
            os.wait()
            #os.spawnv(os.P_NOWAIT, "../boldgen", mylist)
            os.spawnv(os.P_NOWAIT, "../boldgen", mylist)
    
        else:
            os.spawnv(os.P_NOWAIT, "../boldgen", mylist)
            forks = forks+1
    
    for i in range(0,eststat.timepoints):
        runcmd = "fslmaths simseries.nii.gz -sub data/resim%04i.nii.gz -sqr -Tmean " \
                    "data/mse%04i.nii.gz" % (i, i)
        if forks >= PARALLEL:
            os.wait()
            os.spawnvp(os.P_NOWAIT, "fslmaths",runcmd.split());
        else:
            os.spawnvp(os.P_NOWAIT, "fslmaths",runcmd.split());
            forks = forks+1
    
    os.system('fslmerge -t mse.nii.gz data/mse*nii.gz')

mse = image.NiftiImage('mse')
line1= P.plot(mse.data[:, 0,0,0])
P.show()
