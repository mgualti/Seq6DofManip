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

class RlAgentLevel2(RlAgentLevel):

  def __init__(self, level, params):
    '''Initializes agent in the given environment.'''

    RlAgentLevel.__init__(self, level, params)

    # other internal variables
    self.imD = params["imD"][2]
    self.imW = params["imW"][2]    
    self.selD = params["selD"][2]
    self.selW = params["selW"][2]
    self.imDNext = params["imD"][3]
    self.imWNext = params["imW"][3]
    
    # initialization
    self.actionsInHandFrame = self.SampleActions()

  def SampleActions(self):
    '''Samples hand positions in both base frame and image coordinates.'''

    # generate uniform sampling of 2D positions
    
    dx = self.selW / self.outputShape[0][0] / 2.0
    dy = self.selW / self.outputShape[0][1] / 2.0
    dz = self.selD / self.outputShape[0][2] / 2.0
    
    x = linspace(-self.selW / 2.0 + dx, self.selW / 2.0 - dx, self.outputShape[0][0])
    y = linspace(-self.selW / 2.0 + dy, self.selW / 2.0 - dy, self.outputShape[0][1])
    z = linspace(-self.selD / 2.0 + dz, self.selD / 2.0 - dz, self.outputShape[0][2])
    
    y, x, z = meshgrid(y, x, z)
    return stack((x, y, z), axis=3)

  def SenseAndAct(self, hand, prevDesc, cloud, tableHeight, t, unbias):
    '''TODO'''
    
    # generate input image
    targImage = prevDesc.GenerateHeightmap(cloud, tableHeight)
    handImage = zeros((self.imP, self.imP, 0)) if hand is None else hand.image
    timeImage = zeros((self.imP, self.imP, 0)) if not self.includeTime else \
      float(self.tMax - t) / self.tMax * ones((self.imP, self.imP, 1))
    o = concatenate((targImage, handImage, timeImage), axis = 2)
    
    # decide which location in the image to zoom into
    bestIdx, bestValue, epsilon = self.SelectIndexEpsilonGreedy(o, unbias)

    # compose result
    T = copy(prevDesc.T)
    T[0:3, 3] = self.actionsInHandFrame[bestIdx] + prevDesc.center
    desc = HandDescriptor(T, self.imP, self.imDNext, self.imWNext)
    a = bestIdx

    return o, a, desc, bestValue, epsilon