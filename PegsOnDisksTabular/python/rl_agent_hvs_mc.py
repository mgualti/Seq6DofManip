'''An HVS agent where the Q-values are updated according to the MC update rule.'''

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

class RlAgentHvsMc(RlAgentHvs):

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    RlAgentHvs.__init__(self, params)
    
    self.Sums = {}
    self.Counts = {}    

  def UpdateQFunction(self, o, i, r):
    '''Updates the current q-estimates given an episode of experiences.
    - Input o: A list of observation lists, [[o_0, ..., o_L-1]_0, ... [o_l0, ..., o_L-1]_T].
    - Input i: A list of abstract action lists, formatted as above.
    - Input r: A list of rewards, one for each time step, [r_1, ..., r_T].
    - Returns None.
    '''
    
    for t in xrange(self.tMax):
      
      for l in xrange(self.L):
  
        idx = o[t][l] + (i[t][l],)
        
        # discounted sum of rewards from now onward
        Return = 0
        for j in xrange(t, self.tMax):
          Return += self.gamma**(j - t) * r[j]
        
        # update Q
        if idx not in self.Sums:
          self.Sums[idx] = 0.00
          self.Counts[idx] = 0.00
        self.Sums[idx] += Return
        self.Counts[idx] += 1.00
        self.Q[idx] = self.Sums[idx] / self.Counts[idx]