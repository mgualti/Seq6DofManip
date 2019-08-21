'''One sense level of the hierarchical sampling agent.'''

# python
from copy import copy
from time import time
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import any, arange, argmax, argmin, argsort, array, concatenate, eye, exp, hstack, \
  linspace, logical_and, isinf, mean, meshgrid, ones, repeat, reshape, round, sqrt, stack, tile, \
  unique, where, unravel_index, vstack, zeros
# self
from hand_descriptor import HandDescriptor
from rl_agent_level import RlAgentLevel

# AGENT ============================================================================================

class RlAgentLevelGrasp0(RlAgentLevel):

  def __init__(self, params):
    '''Initializes agent in the given environment.'''

    RlAgentLevel.__init__(self, 0, True, False, params)

    # other internal variables
    self.imD = params["imD"][0]
    self.imW = params["imW"][0]
    self.selD = params["selD"][0]
    self.selW = params["selW"][0]  
    self.imDNext = params["imD"][1]
    self.imWNext = params["imW"][1]
    
    self.qMax = 1.0
    self.qMin = -1.0
    
    # initialization
    self.SetInitialDescriptor(zeros(3))
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
    
    prevDesc = self.initDesc
    
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
    
    prevDesc = self.initDesc
    
    # generate input image
    targImage = prevDesc.GenerateHeightmap(cloud, tableHeight)
    handImage = zeros((self.imP, self.imP, 0)) if hand is None else hand.image
    timeImage = zeros((self.imP, self.imP, 0)) if not self.includeTime else \
      float(self.tMax - t) / self.tMax * ones((self.imP, self.imP, 1))
    o = concatenate((targImage, handImage, timeImage), axis = 2)

    # evaluate action-values and sample n of them
    values = self.EvaluateActions(o).flatten()
    flatActionIdxs = argsort(-values)[:len(prevDescs)]
    values = values[flatActionIdxs]
    
    # convert values into probabilities (of separate Bernoulli random variables)
    values[values > self.qMax] = self.qMax
    values[values < self.qMin] = self.qMin
    probabilities = (values - self.qMin) / (self.qMax - self.qMin)
    
    # create a descriptor for each sampled action
    descriptors = []
    for i, flatIdx in enumerate(flatActionIdxs):
      idx = unravel_index(flatIdx, self.outputShape)
      T = copy(prevDesc.T)
      T[0:3, 3] = self.actionsInHandFrame[idx] + prevDesc.center
      descriptors.append(HandDescriptor(T, self.imP, self.imDNext, self.imWNext))
      if self.plotImages and i == 0:
        print("Probability: {}.".format(probabilities[i]))
        self.PlotImages(o, idx, descriptors[-1])

    return descriptors, probabilities

  def SetInitialDescriptor(self, offset):
    '''Sets the center of the initial descriptor to the provided center.'''
    
    T = eye(4)
    T[0:3, 3] = array([offset[0], offset[1], offset[2] + self.imD / 2.0])
    self.initDesc = HandDescriptor(T, self.imP, self.imD, self.imW)
    self.initDesc.imW = self.imW
    self.initDesc.imD = self.imD