'''An HVS agent where the Q-values are updated according to the Q-learning update rule.'''

# python
import os
import pickle
from time import time
from copy import copy
# scipy
from numpy.random import rand, randint
from numpy import argmax, array, ceil, cos, cross, delete, dot, eye, hstack, log2, nonzero, ones, \
  pi, ravel_multi_index, sin, tile, vstack, unravel_index, where, zeros
# drawing
from matplotlib import pyplot
from skimage.draw import ellipse_perimeter, line
# self
from rl_agent_hvs import RlAgentHvs

# AGENT ============================================================================================

class RlAgentHvsQl(RlAgentHvs):

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    RlAgentHvs.__init__(self, params)

  def UpdateQFunction(self, o, i, r, oo):
    '''Updates the current q-estimates given an (overt) time step of experiences.
    - Input o: A list of observations, [o_0, ..., o_L-1].
    - Input i: A list of abstract actions, [i_0, ..., i_L-1].
    - Input r: The (scalar) reward received after taking this action from this state.
    - Input oo: List of observations in the next (overt) time step, [oo_0, ..., oo_L-1].
    - Returns None.
    '''
    
    for l in xrange(self.L):

      idx = o[l] + (i[l],)

      if l != self.L - 1:
        ll = l + 1
        nextObservation = o[ll]
        rr = 0
      else:
        ll = 0
        nextObservation = None if oo is None else oo[ll]
        rr = r
      
      if nextObservation is not None:
        # take max over next actions
        tt = nextObservation[-1]
        bestValue = -float('inf'); nextAction = None
        for ii in xrange(8):
          jdx = nextObservation + (ii,)
          if jdx not in self.Q: self.Q[jdx] = self.initQ[tt]
          if self.Q[jdx] > bestValue:
            bestValue = self.Q[jdx]
            nextAction = ii
        jdx = nextObservation + (nextAction, )
      else:
        # terminal state
        jdx = None
        
      # update Q
      self.Q[idx] = (1.0 - self.alpha) * self.Q[idx] + self.alpha * (rr + self.gamma * self.Q[jdx])