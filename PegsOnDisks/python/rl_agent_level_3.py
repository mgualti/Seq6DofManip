'''One sense level of the HSA agent.'''

# python
from copy import copy
from time import time
# scipy
from numpy.linalg import norm
from numpy.random import choice, rand, randint, uniform
from numpy import any, argmax, argmin, argsort, array, concatenate, eye, exp, hstack, linspace, \
  logical_and, isinf, meshgrid, ones, pi, repeat, reshape, round, sqrt, stack, tile, where, \
  unravel_index, vstack, zeros
# openrave
import openravepy
# self
from hand_descriptor import HandDescriptor
from rl_agent_level import RlAgentLevel

# AGENT ============================================================================================

class RlAgentLevel3(RlAgentLevel):

  def __init__(self, level, params):
    '''TODO'''

    RlAgentLevel.__init__(self, level, params)

    # other internal variables
    self.imD = params["imD"][3]
    self.imW = params["imW"][3]
    self.imDNext = params["imD"][4]
    self.imWNext = params["imW"][4]
    
    # initialization
    self.actionsGrasp = self.SampleActions(True)
    self.actionsPlace = self.SampleActions(False)

  def SampleActions(self, forGrasp):
    '''TODO'''
    
    theta = linspace(-pi / 2.0, pi / 2.0, self.nGraspOrientations) if forGrasp else \
            linspace(-pi / 2.0, pi / 2.0, self.nPlaceOrientations)
    axis = array([0.0, 0.0, 1.0]) if forGrasp else array([0.0, 1.0, 0.0])
    
    rotations = []
    for t in theta:
      R = openravepy.matrixFromAxisAngle(axis, t)[0:3, 0:3]
      rotations.append(R)    
    
    return rotations

  def SenseAndAct(self, hand, prevDesc, cloud, tableHeight, t, unbias):
    '''TODO'''
    
    # generate input image
    targImage = prevDesc.GenerateHeightmap(cloud, tableHeight)
    handImage = zeros((self.imP, self.imP, 0), dtype="float32") if hand is None else hand.image
    timeImage = zeros((self.imP, self.imP, 0), dtype="float32") if not self.includeTime else \
      float(self.tMax - t) / self.tMax * ones((self.imP, self.imP, 1), dtype="float32")
    o = concatenate((targImage, handImage, timeImage), axis = 2)
    
    # decide which location in the image to zoom into
    bestIdx, bestValue, epsilon = self.SelectIndexEpsilonGreedy(o, unbias)

    # compose result
    T = copy(prevDesc.T)
    actions = self.actionsGrasp if hand is None else self.actionsPlace
    T[0:3, 0:3] = actions[bestIdx]
    desc = HandDescriptor(T, self.imP, self.imDNext, self.imWNext)
    a = bestIdx

    return o, a, desc, bestValue, epsilon