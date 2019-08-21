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

class RlAgentLevelPlace2(RlAgentLevel):

  def __init__(self, params):
    '''Initializes agent in the given environment.'''

    RlAgentLevel.__init__(self, 2, False, False, params)

    # other internal variables
    self.imD = params["imD"][2]
    self.imW = params["imW"][2]    
    self.selD = params["selD"][2]
    self.selW = params["selW"][2]
    self.imDNext = params["imD"][3]
    self.imWNext = params["imW"][3]
    
    self.qMax = 1.0
    self.qMin = 0.0
    
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
    
  def SenseAndActMultiple(self, hand, prevDescs, prevProbs, cloud, tableHeight, t):
    '''TODO'''
    
    # generate input image
    observations = []
    for prevDesc in prevDescs:
      targImage = prevDesc.GenerateHeightmap(cloud, tableHeight)
      handImage = zeros((self.imP, self.imP, 0)) if hand is None else hand.image
      timeImage = zeros((self.imP, self.imP, 0)) if not self.includeTime else \
        float(self.tMax - t) / self.tMax * ones((self.imP, self.imP, 1))
      o = concatenate((targImage, handImage, timeImage), axis = 2)
      observations.append(o)
    
    # evaluate action-values and sample n of them
    valuesBatch = self.EvaluateActionsMultiple(observations).flatten()
    flatActionIdxs = argsort(-valuesBatch)[:len(prevDescs)]
    values = valuesBatch[flatActionIdxs]
    
    # convert values into probabilities (of separate Bernoulli random variables)
    values[values > self.qMax] = self.qMax
    values[values < self.qMin] = self.qMin
    probabilities = (values - self.qMin) / (self.qMax - self.qMin)
    
    # create a descriptor for each sampled action
    descriptors = []
    for i, flatIdx in enumerate(flatActionIdxs):
      idx = unravel_index(flatIdx, (len(prevDescs), ) + self.outputShape)
      T = copy(prevDescs[idx[0]].T)
      T[0:3, 3] += self.actionsInHandFrame[idx[1:]]
      descriptors.append(HandDescriptor(T, self.imP, self.imDNext, self.imWNext))
      probabilities[i] *= prevProbs[idx[0]]

    return descriptors, probabilities