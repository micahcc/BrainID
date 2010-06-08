#!/usr/bin/python
import pylab as P
import nibabel 
import sys

paramnames = [ "TAU_0  " , "ALPHA  " , "E_0    " , "V_0    " , "TAU_S  " , "TAU_F  " , "EPSILON" ]

#Begin Main, main
time = 0
if len(sys.argv) == 3:
    try:
        time = float(sys.argv[2])
    except:
        time = 0
elif len(sys.argv) != 2:
    print "Usage: ", sys.argv[0], "<InDir> [discrete time]"
    print "Looks in Dir for: "
    print  "histogram.nii.gz"
    sys.exit(-1);

for param in range(len(paramnames)):
    print paramnames[param], "!!"
    P.subplot(3, 3, param+1)
    histimg = nibabel.load(sys.argv[1] + "histogram.nii.gz")
    mywidth= (histimg.get_data()[0,0,0,time, param,-2] - histimg.get_data()[0,0,0,time, param,-3])/ \
                (histimg.get_header()['dim'][6]-3)
    print mywidth
    lpoints = [mywidth*i+histimg.get_data()[0,0,0,time, param,-3] for i in range(histimg.get_header()['dim'][6]-3)]
    print lpoints
    P.bar(lpoints, histimg.get_data()[0,0,0,time, param,:-3], width=mywidth/2.)
    P.xlabel(paramnames[param])
P.show()
