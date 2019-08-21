'''Same as the HVS agent but with limited sensing capabilities.'''

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

class RlAgentHvsLimitedSensor(RlAgentHvs):

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''
    
    RlAgentHvs.__init__(self, params)

  def GetActionsAndObservations(self, s, epsilon):
    '''Gets a list of actions, abstract actions, and observations for each level in the sense sequence.
    - Input s: The current state = (pegs, holes, time).
    - Returns a: The underlying action (world coordinates flattened index).
    - Returns i: The abstract actions, one for each sense level, each of which is an index
      indicating which cell in the observation was selected.
    - Returns o: The  observations, one for each sense level l, each of which is the tuple from
      GetObservation.
    '''
    
    # reached terminal state
    if s is None:
      return None, None, None
    
    # decompose state information
    pegs = s[0:self.n]; disks = s[self.n:2*self.n]; t = s[-1]
    
    # !!! Not ignoring peg-disks !!!
    # The sensor is limited because it cannot distinguish pegs/disks from peg-disks.
    
    # compute coordinates for pegs and disks
    pegCoords = []; diskCoords = []; h = False
    for i in xrange(len(pegs)):
      if pegs[i] == self.endEffectorIdx:
        h = True
      else:
        pegCoords.append(unravel_index(pegs[i], self.worldSize))
      diskCoords.append(unravel_index(disks[i], self.worldSize))    
    
    # initialize outputs
    a = (0, 0, 0); i = []; o = []

    # take best action, breaking ties randomly
    for l in xrange(self.L):
      observation = self.GetObservation(pegCoords, diskCoords, h, l, t, a)
      if rand() < epsilon:
        abstractAction = randint(8)
        idx = observation + (abstractAction,)
        if idx not in self.Q: self.Q[idx] = self.initQ[t]
      else:
        bestValue = -float('inf'); abstractActions = None
        for abstractAction in xrange(8):
          idx = observation + (abstractAction,)
          if idx not in self.Q: self.Q[idx] = self.initQ[t]
          if self.Q[idx] > bestValue:
            bestValue = self.Q[idx]
            abstractActions = [abstractAction]
          elif self.Q[idx] == bestValue:
            abstractActions.append(abstractAction)
        # break ties randomly
        abstractAction = abstractActions[randint(len(abstractActions))]
      # compute new sensor location and append observation and abstract action to list
      a = self.GetAction(a, l, abstractAction)
      o.append(observation); i.append(abstractAction)
      
      # visualization
      if self.plotImages:
        print("Best value: {}".format(bestValue))
        action = ravel_multi_index(a, self.worldSize)
        self.PlotImages(s, action, observation, abstractAction)

    return ravel_multi_index(a, self.worldSize), i, o