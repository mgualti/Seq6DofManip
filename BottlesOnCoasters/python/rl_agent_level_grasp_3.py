'''One sense level of the HSA agent.'''

# python
from copy import copy
from time import time
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import any, arange, argmax, argmin, argsort, array, concatenate, eye, exp, hstack, \
  linspace, logical_and, isinf, meshgrid, ones, pi, repeat, reshape, round, sqrt, stack, tile, \
  where, unravel_index, vstack, zeros
# openrave
import openravepy
# self
from hand_descriptor import HandDescriptor
from rl_agent_level import RlAgentLevel

# AGENT ============================================================================================

class RlAgentLevelGrasp3(RlAgentLevel):

  def __init__(self, params):
    '''TODO'''

    RlAgentLevel.__init__(self, 3, True, True, params)

    # other internal variables
    self.imD = params["imD"][3]
    self.imW = params["imW"][3]
    self.imDNext = params["imD"][4]
    self.imWNext = params["imW"][4]
    
    self.qMax = 1.0
    self.qMin = -1.0
    
    # initialization
    self.actionsInHandFrame = self.SampleActions()

  def SampleActions(self):
    '''TODO'''
    
    theta = linspace(-pi / 2.0, pi / 2.0, self.nGraspOrientations)
    axis = array([0.0, 0.0, 1.0])
    
    rotations = []
    for t in theta:
      R = openravepy.matrixFromAxisAngle(axis, t)[0:3, 0:3]
      rotations.append(R)    
    
    return rotations

  def SenseAndAct(self, hand, prevDesc, cloud, tableHeight, t, unbias):
    '''TODO'''
    
    # generate input image
    targImage = prevDesc.GenerateHeightmap(cloud, tableHeight)
    handImage = zeros((self.imP, self.imP, 0), dtype='float32') if hand is None else hand.image
    timeImage = zeros((self.imP, self.imP, 0), dtype='float32') if not self.includeTime else \
      float(self.tMax - t) / self.tMax * ones((self.imP, self.imP, 1))
    o = concatenate((targImage, handImage, timeImage), axis = 2)
    
    # decide which location in the image to zoom into
    bestIdx, bestValue, epsilon = self.SelectIndexEpsilonGreedy(o, unbias)

    # compose result
    T = copy(prevDesc.T)
    T[0:3, 0:3] = self.actionsInHandFrame[bestIdx]
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
      T[0:3, 0:3] = self.actionsInHandFrame[idx[1]]
      descriptors.append(HandDescriptor(T, self.imP, self.imDNext, self.imWNext))
      probabilities[i] *= prevProbs[idx[0]]

    return descriptors, probabilities