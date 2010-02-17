#!/usr/bin/python
from math import sqrt
import sys
import getopt
import pylab as P
from nifti import *

def main(argv):                         
    truebold = "bold"
    truestate = "state"
    estbold = "particle-bold"
    eststate = "particle-state"
    saveloc = ""
    try:                                
        opts, args = getopt.getopt(argv, "hb:s:y:x:p:", ["help", "bold=", 
                    "state=", "estbold=", "eststate=", "prefix="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage();
            sys.exit();
        elif opt in ("-b", "--bold"):
            truebold = arg;
        elif opt in ("-s", "--state"):
            truestate = arg;
        elif opt in ("-x", "--eststate"):
            eststate = arg;
        elif opt in ("-y", "--estbold"):
            estbold = arg;
        elif opt in ("-p", "--prefix"):
            saveloc = arg

    truebold = image.NiftiImage(truebold)
    try: 
        truestate = image.NiftiImage(truestate)
    except  getopt.GetoptError:
        print "Hoops"
    
    estbold = image.NiftiImage(estbold)
    eststate = image.NiftiImage(eststate)
    print eststate
    print estbold
    
    leg = []
    time = range(0, 1199, 2)
    
    for iter in range(0,truebold.extent[0]):
        P.plot(time, truebold.data[:, 0,0,iter])
        leg = leg + ["Simulated BOLD"]
    
    for iter in range(0,estbold.extent[0]):
        bold = estbold.data[:, 0, 0, iter]
        error = estbold.data[:, 1, 0, iter]
        for errno in range(0, len(error)):
            error[errno] = 2*sqrt(error[errno])
        if(estbold.extent[2] == 2):
            P.errorbar(time, bold, yerr=error, fmt='r', elinewidth=.5, ecolor='g')
        else:
            P.plot(time, estbold.data[:, 0,0,iter], 'b')
        leg = leg + ["Estimated BOLD"]
        
    P.xlabel("Time, Seconds")
    P.ylabel("BOLD Signal %")
    if not saveloc:
        P.show()
    else:
        P.savefig(saveloc + "-bold.jpg")
    
    leg = []
    #names = ['T0', 'alpha', 'E_0', 'V0', 'Ts', 'Tf', 'epsilon', 'Vt', 'Qt', 'St', 'Ft']
    names = ['', '', '', 'V0', 'Ts', 'Tf', '', '', '', 'St', '']
    j = 0
    for i in range(0,eststate.extent[1]):
        if names[i] == '':
            continue
        j  = j + 1
        P.subplot(4, 1, j)
        P.ylabel(names[i])
        stat = eststate.data[:, 0, i, 0]
        error = eststate.data[:, 1, i, 0]
        for errno in range(0, len(error)):
            error[errno] = 2*sqrt(error[errno])
            
        if(eststate.extent[2] == 2):
            P.errorbar(time, stat, yerr=error, fmt='r', elinewidth=.5, ecolor='g')
        else:
            P.plot(time, eststate.data[:, 0, i, 0],'b')
    
    j=0
    if(truestate != "state"):
        for i in range(0,truestate.extent[1]):
            if names[i] == '':
                continue
            j = j + 1
            P.subplot(4, 1, j)
            P.plot(time, truestate.data[:, 0, i, 0])
    
#    regions = (eststate.extent[1]-4)/7
#    for i in range(0, regions):
#    #    leg = leg + [truestat.filename+":"+str(i)]
#        leg = leg + [eststat.filename+":"+str(i)]
        
#    P.subplot(4,1, 11)
#    P.legend(leg)
    P.xlabel("Time, Seconds")
    if not saveloc:
        P.show()
    else:
        P.savefig(saveloc + "-state.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
