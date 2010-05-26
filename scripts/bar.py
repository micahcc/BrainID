#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

basecolor = [.9, .8, .8]

class histelem:
    weight  = 0
    size = 0

    def __init__(self, level, size):
        self.weight= level
        self.size = size

    def __str__(self):
        return "(" + str(self.weight) + ":" + str(self.size) + ")"
    
class histo:
    elems = []

    def __init__(self, newelems):
        self.elems = []
        total = float(sum([elem[0] for elem in newelems]))
        
        if newelems[0][1] > 0:
            self.elems.append(histelem(-1, newelems[0][1]))
        if newelems[-1][2] < 0:
            self.elems.append(histelem(-1, newelems[-1][2]))

        for elem in newelems:
            if elem[2] < 0 and elem[1] < 0:
                self.elems.append(histelem(elem[0]/total, elem[1]))
            elif elem[2] > 0 and elem[1] > 0:
                self.elems.append(histelem(elem[0]/total, elem[2]))
            else:
                self.elems.append(histelem(elem[0]/total, elem[1]))
                self.elems.append(histelem(elem[0]/total, elem[2]))
        
        self.elems = sorted(self.elems, key = lambda var: abs(var.size), reverse=True)
#        print str(self)

    def plot(self, pos, Width):
#        print "Plotting"
        for elem in self.elems:
            if elem.weight == -1:
                c = (1,1,1)
            else:
                c = tuple([cpos*(1-elem.weight) for cpos in basecolor])
            plt.bar(pos, elem.size, color=c, edgecolor="w", width=Width)
    
    def __str__(self):
        out = ""
        for elem in self.elems:
            out = out + " " + str(elem)
        return out

def plothisto(ts, TR):
    if not isinstance(ts, type([])) or not isinstance(ts[0], histo):
        raise "Wrong type"
    """note for every time, everything needs to add up to the same value"""
    xax = [i*TR for i in range(len(ts))]
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
