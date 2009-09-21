#!/usr/bin/python
import sys
import getopt
import pylab as P
from nifti import *

def usage():
    print("-b", "--bold")
    print("-s", "--state")
    print("-x", "--eststate")
    print("-y", "--estbold")
    print("-p", "--prefix")

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
    
    for iter in range(0,truebold.extent[0]):
        P.plot(truebold.data[:, 0,0,iter])
        leg = leg + [truebold.filename+":"+str(iter)]
    
    for iter in range(0,estbold.extent[0]):
        if(estbold.extent[2] == 2):
            P.errorbar(range(0, estbold.extent[3]), estbold.data[:, 0,0,iter],
                        yerr=estbold.data[:, 1,0,iter])
        else:
            P.plot(estbold.data[:, 0,0,iter])
        leg = leg + [estbold.filename+":"+str(iter)]
    P.legend(leg)
    if not saveloc:
        P.show()
    else:
        P.savefig(saveloc + "-bold.jpg")
    
    leg = []
    names = ['T0', 'alpha', 'E_0', 'V0', 'Ts', 'Tf', 'epsilon', 'Vt', 'Qt', 'St', 'Ft']
    for i in range(0,eststate.extent[1]):
        if i < 4:
            P.subplot(6, 2, i+1)
            P.ylabel(names[i])
        else:
            P.subplot(6, 2, (i-4)%7+4+1)
            P.ylabel(names[(i-4)%7+4])
        if(eststate.extent[2] == 2):
            P.errorbar(range(0,eststate.extent[3]), eststate.data[:, 0, i, 0], yerr=eststate.data[:, 1, i, 0])
        else:
            P.plot(range(0,eststate.extent[3]), eststate.data[:, 0, i, 0])
    
    if(truestate != "state"):
        for i in range(0,truestate.extent[1]):
            if i < 4:
                P.subplot(6, 2, i+1)
            else:
                P.subplot(6, 2, (i-4)%7+4+1)
            P.plot(range(0,truestate.extent[3]), truestate.data[:, 0, i, 0])
    
#    regions = (eststate.extent[1]-4)/7
#    for i in range(0, regions):
#    #    leg = leg + [truestat.filename+":"+str(i)]
#        leg = leg + [eststat.filename+":"+str(i)]
        
    P.subplot(6,2, 11)
#    P.legend(leg)
    if not saveloc:
        P.show()
    else:
        P.savefig(saveloc + "-state.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
