#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

basecolor = [.9, 1, .9]

class histelem:
    weight  = 0
    seq = (0,0)

    def __init__(self, level, start, size):
        self.weight= level
        self.seq = (start, size)

    def __str__(self):
        return "(" + str(self.weight) + ":" + str(self.seq[0]) + ":" + str(self.seq[1]) + ")"
    
class histo:
    elems = []

    def __init__(self, newelems):
        self.elems = []
        total = float(sum([elem[0] for elem in newelems]))
        
        for elem in newelems:
            self.elems.append(histelem(elem[0]/total, elem[1], elem[2]-elem[1]))
        
        self.elems = sorted(self.elems, key = lambda var: var.seq[0])
#        print str(self)

    def plot(self, pos, Width):
#        print "Plotting"
#        pred = max([comp.weight for comp in self.elems])
        for elem in self.elems:
#            if elem.weight == pred:
#                c = [0, 1, 0]
#            else:
            c = tuple([cpos*(1-elem.weight) for cpos in basecolor])
            plt.broken_barh([(pos, Width)], elem.seq, color=c, edgecolor=c)
    
    def __str__(self):
        out = ""
        for elem in self.elems:
            out = out + " " + str(elem)
        return out

def plothisto(ts, TR):
    if not isinstance(ts, type([])):
        print ts
        raise "Wrong type"
    if not isinstance(ts[0], histo):
        print ts[0]
        raise "Wrong type"

    """note for every time, everything needs to add up to the same value"""
    xax = [i*TR-3.*TR/2. for i in range(len(ts))]
    print xax
    for i in range(0, len(ts)):
#        print ts[i]
        ts[i].plot(xax[i], TR)

#hist1 = [histo([[12, -1, 3], [2, 3,4], [1, 4,5]]),
#        histo([[ 9, 2,3],  [3, 3,4], [ 4, 4,5]]),
#        histo([[ 6, 2.5,3],[2, 3,4], [12, 4,5]]),
#        histo([[ 6, 1,2],[2, 3,4], [12, 4,5]])]
#
#plothisto(hist1, 2)
#plt.show()
