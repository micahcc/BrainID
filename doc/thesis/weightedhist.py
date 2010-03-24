import pylab
import numpy
from scipy.stats import norm
import random

def rolldice(num):
    sum = 0
    for i in range(0, num):
        sum = sum + random.randrange(1, 7)
    return sum

def two_die(total):
    if total < 2 or total > 12:
        return 0

    density = range(1, 7)
    density.extend(range(5,0,-1))
    return density[total-2]

def test():
    bin = [x + 1.5 for x in range(0, 12)]
    pointsv = [rolldice(2) for i in range(0, 50000)]
    pylab.hist(pointsv, bin);
    pylab.show()

    weightv = [1./two_die(point) for point in pointsv]
    pylab.hist(pointsv, bin, weights=weightv)
    pylab.show()

def test2():
    pointsv = pylab.randn(50000);
    pylab.hist(pointsv);
    pylab.show()

    weightv = [1./norm.pdf(point) for point in pointsv]
    pylab.hist(pointsv, 30, weights=weightv)
    pylab.show()

test2()
