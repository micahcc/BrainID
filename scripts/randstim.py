#!/usr/bin/python

from optparse import OptionParser
import random

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
            help="write stimulus sequence to FILE", metavar="FILE")
parser.add_option("-c", "--chance", dest="chance", default=.01, type="float",
            help="Chance of high signal")
parser.add_option("-r", "--resolution", dest="resolution", default=.1, type="float",
            help="Time between signals")
parser.add_option("-u", "--ulevel", dest="ulevel", default=1.0, type="float",
            help="\"High\" level")
parser.add_option("-l", "--level", dest="level", default=0.0, type="float",
            help="Default level")
parser.add_option("-s", "--start", dest="start", default=0.0, type="float",
            help="Time to start (at set to 0 before this)")
parser.add_option("-p", "--pad", dest="pad", default=0.0, type="float",
            help="amount of time before end to stop allowing non-defualt stimulus")
parser.add_option("-e", "--end", dest="end", default=1024, type="float",
            help="Ending time")
(options, args) = parser.parse_args()

print "Chance: ", options.chance
print "Resolution: ", options.resolution
print "File: ", options.filename
file = open(options.filename, "w");
for i in range(0,int(options.end/options.resolution)):
    t = i*options.resolution
    rand = random.random()
    if t < options.start or t > (options.end-options.pad) or rand > options.chance:
        file.write(str(t))
        file.write(" " + str(options.level) + "\n")
    else:
        file.write(str(t))
        file.write(" " + str(options.ulevel) + "\n")

        
