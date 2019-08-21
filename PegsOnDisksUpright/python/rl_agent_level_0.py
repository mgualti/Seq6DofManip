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
from rl_agent_level import RlAgentLevel

# AGENT ============================================================================================

class RlAgentLevel0(RlAgentLevel):

  def __init__(self, level, params):
    '''Initializes agent in the given environment.'''

    RlAgentLevel.__init__(self, level, params)

    # other internal variables
    self.SetInitialDescriptor()
    self.imW = self.initDesc.imW
    self.imD = self.initDesc.imD
    self.selD = self.initDesc.imD
    self.selW = self.initDesc.imW
    
    # initialization
    self.actionsInHandFrame = self.SampleActions()

  def SampleActions(self):
    '''Samples hand positions in both base frame and image coordinates.'''

    # generate uniform sampling of 2D positions
    
    dx = self.selW / self.outputShape[0] / 2.0
    dy = self.selW / self.outputShape[1] / 2.0
    dz = self.selD / self.outputShape[2] / 2.0
    
    x = linspace(-self.selW / 2.0 + dx, self.selW / 2.0 - dx, self.outputShape[0])
    y = linspace(-self.selW / 2.0 + dy, self.selW / 2.0 - dy, self.outputShape[1])
    z = linspace(-self.selD / 2.0 + dz, self.selD / 2.0 - dz, self.outputShape[2])
    
    y, x, z = meshgrid(y, x, z)
    return stack((x, y, z), axis=3)

  def SenseAndAct(self, hand, prevDesc, t, rlEnv, unbias):
    '''TODO'''
    
    prevDesc = self.initDesc
    
    # generate input image
    targImage = prevDesc.GenerateHeightmap(rlEnv)
    handImage = zeros((self.imP, self.imP, 0)) if hand is None else hand.image
    o = concatenate((targImage, handImage), axis = 2)

    # decide which location in the image to zoom into
    bestIdx, bestValue, epsilon = self.SelectIndexEpsilonGreedy(o, unbias)

    # compose result
    T = copy(prevDesc.T)
    T[0:3, 3] = self.actionsInHandFrame[bestIdx] + prevDesc.center
    desc = HandDescriptor(T, self.params)
    a = bestIdx

    return o, a, desc, bestValue, epsilon

  def SetInitialDescriptor(self):
    '''Sets the center of the initial descriptor to the provided center.'''
    
    W = 0.36
    D = self.params["objHeight"][1] + 0.0375  
    
    T = eye(4)
    T[0:3, 3] = array([0, 0, D / 2.0])
    self.initDesc = HandDescriptor(T, self.params)
    self.initDesc.imW = W
    self.initDesc.imD = D    