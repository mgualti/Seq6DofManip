'''One sense level of the hierarchical sampling agent.'''

# python
from copy import copy
from time import time
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import any, argmax, argmin, argsort, array, concatenate, eye, exp, hstack, linspace, \
  logical_and, isinf, meshgrid, ones, repeat, reshape, round, sqrt, stack, tile, where, \
  unravel_index, vstack, zeros
# self
from hand_descriptor import HandDescriptor
from rl_agent_lookahead_level import RlAgentLookaheadLevel

# AGENT ============================================================================================

class RlAgentLookaheadLevel1(RlAgentLookaheadLevel):

  def __init__(self, level, params):
    '''Initializes agent in the given environment.'''

    RlAgentLookaheadLevel.__init__(self, level, params)

    # other internal variables
    self.imW = params["imW"]
    self.imD = params["imD"]
    self.selW = 0.09
    self.selD = 0.09
    
    # initialization
    self.actionsInHandFrame = self.SampleActions()

  def SampleActions(self):
    '''Samples hand positions in both base frame and image coordinates.'''

    # generate uniform sampling of 2D positions
    
    dx = self.selW / self.actionSpaceSize[0] / 2.0
    dy = self.selW / self.actionSpaceSize[1] / 2.0
    dz = self.selD / self.actionSpaceSize[2] / 2.0
    
    x = linspace(-self.selW / 2.0 + dx, self.selW / 2.0 - dx, self.actionSpaceSize[0])
    y = linspace(-self.selW / 2.0 + dy, self.selW / 2.0 - dy, self.actionSpaceSize[1])
    z = linspace(-self.selD / 2.0 + dz, self.selD / 2.0 - dz, self.actionSpaceSize[2])
    
    y, x, z = meshgrid(y, x, z)
    return stack((x, y, z), axis=3)

  def SenseAndAct(self, hand, prevDesc, t, rlEnv, unbias):
    '''TODO'''
    
    handImage = zeros((self.imP, self.imP, 0)) if hand is None else hand.image
    
    # generate candidate descriptors
    descs = []
    for i in xrange(self.actionSpaceSize[0]):
      for j in xrange(self.actionSpaceSize[1]):
        for k in xrange(self.actionSpaceSize[2]):
          T = copy(prevDesc.T)
          T[0:3, 3] = self.actionsInHandFrame[(i, j, k)] + prevDesc.center
          descs.append(HandDescriptor(T, self.params))

    # decide which location in the image to zoom into
    bestIdx, bestValue, epsilon = self.SelectIndexEpsilonGreedy(descs, handImage, unbias, rlEnv)

    # compose result
    desc = descs[bestIdx]    
    o = concatenate((desc.image, handImage), axis = 2)

    return o, desc, bestValue, epsilon