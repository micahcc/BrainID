import sys
import nibabel
import time
import pylab
import numpy
import math

TAU_0, ALPHA, E_0, V_0, TAU_S, TAU_F, EPSILON, A_1, A_2 = range(0, 9)
params = (8.38, .189, .635, 1.49e-2, 4.98, 8.31, 0.069, 3.4, 1)

class State:
    F = 1
    S = 0
    V = 1
    Q = 1

    def __init__(self):
        self.F = 1
        self.S = 0
        self.V = 1
        self.Q = 1

def readout(state, params):
    return params[V_0]*(params[A_1]*(1-state.Q)-params[A_2]*(1-state.V))

def transition(state, params, delta, stim):
    change = State()
    change.S = params[EPSILON]*stim - state.S/params[TAU_S] - \
                (state.F - 1.)/params[TAU_F]
    change.F = state.S
    change.V = (state.F - pow(state.V, 1./params[ALPHA]))/params[TAU_0]
    change.Q = (state.F*(1.-pow(1.-params[E_0],1./state.F))/params[E_0] - \
                state.Q/pow(state.V, 1.-1./params[ALPHA]))/params[TAU_0]
    state.S += change.S*delta
    state.F += change.F*delta
    state.V += change.V*delta
    state.Q += change.Q*delta
    return state

len = 7000 #30 seconds
T = [i/100. for i in range(0, len)]
stim = [t < 3.4 and t > 3.3 for t in T]
statevars = State()
bold = [readout(transition(statevars, params, .01, u), params) for u in stim]

pylab.plot(T, bold)
pylab.show()
