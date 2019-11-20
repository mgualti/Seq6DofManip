'''Class and utilities for an HSA agent.'''

# python
import gc
import os
import shutil
from copy import copy
from time import time
# scipy
from matplotlib import pyplot
from numpy.random import choice, permutation, rand, randint
from numpy import argmax, arange, array, cumsum, exp, fliplr, flipud, logical_not, isinf, mean, \
  ones, pi, prod, reshape, stack, where, zeros
# tensorflow
import tensorflow
from tensorflow import keras
# self
from rl_agent_level_grasp_0 import RlAgentLevelGrasp0
from rl_agent_level_grasp_1 import RlAgentLevelGrasp1
from rl_agent_level_grasp_2 import RlAgentLevelGrasp2
from rl_agent_level_grasp_3 import RlAgentLevelGrasp3
from rl_agent_level_place_0 import RlAgentLevelPlace0
from rl_agent_level_place_1 import RlAgentLevelPlace1
from rl_agent_level_place_2 import RlAgentLevelPlace2
from rl_agent_level_place_3 import RlAgentLevelPlace3

# AGENT ============================================================================================

class RlAgent():

  def __init__(self, params):
    '''Initializes a hierarchical volume sampling (HVS) agent.
    - Input params: Hyperparameters dictionary.
    '''

    # parameters
    
    self.tMax = params["tMax"]
    self.deviceId = params["deviceId"]
    self.gamma = params["gamma"]
    self.plotImages = params["plotImages"]

    # internal variables

    self.updateRule = None

    # decide which device to use

    if self.deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.deviceId)

    # initialize a sense agent at each level
    
    self.nLevels = 4
    
    self.graspAgents = []
    self.graspAgents.append(RlAgentLevelGrasp0(params))
    self.graspAgents.append(RlAgentLevelGrasp1(params))
    self.graspAgents.append(RlAgentLevelGrasp2(params))
    self.graspAgents.append(RlAgentLevelGrasp3(params))
    
    self.placeAgents = []
    self.placeAgents.append(RlAgentLevelPlace0(params))
    self.placeAgents.append(RlAgentLevelPlace1(params))
    self.placeAgents.append(RlAgentLevelPlace2(params))
    self.placeAgents.append(RlAgentLevelPlace3(params))
    
    self.agents = self.graspAgents + self.placeAgents

  def AddExperienceMonteCarlo(self, observations, actions, rewards, isGrasp):
    '''Adds an experience, everything needed for a Monte Carlo update, to the agent's replay database.
    - Input observations: List of lists, [[o_1, ..., o_l]_0, ..., [o_1, ..., o_l]_T], an observation
      at each time step and level.
    - Input actions: An action at each time step and level.
    - Input rewards: A reward, one for each time step (from t=0 to t=T).
    - Returns None.
    '''
    
    for t in xrange(self.tMax):
      
      agents = self.graspAgents if isGrasp[t] else self.placeAgents      
      
      Return = 0
      for i in xrange(t, self.tMax):
        Return += self.gamma**(i - t) * rewards[i]
      
      for l in xrange(self.nLevels):

        state = observations[t][l]
        action = actions[t][l]

        experience = [state, action, Return]
        agents[l].AddExperience(experience)

  def GetNumberOfExperiences(self):
    '''Returns the number of experiences in the experience replay database at each level.'''
    
    nExperiences = 0
    for agent in self.agents:
      nExperiences += agent.GetNumberOfExperiences()
    return nExperiences

  def LoadExperienceDatabase(self):
    '''Loads an experience replay database for each agent from file into memory.'''

    for agent in self.agents:
      agent.LoadExperienceDatabase()

  def LoadQFunction(self):
    '''Loads a tensorflow model and network weights from file into memory.'''

    for agent in self.agents:
      agent.LoadQFunction()
      
  def SaveExperienceDatabase(self):
    '''Saves the experience replay databases to file.'''

    for agent in self.agents:
      agent.SaveExperienceDatabase()

  def SaveQFunction(self):
    '''Saves the tensorflow models and weights to file.'''

    for agent in self.agents:
      agent.SaveQFunction()

  def SenseAndAct(self, holdingDescriptor, cloud, t, rlEnv, unbias):
    '''TODO'''

    agents = self.graspAgents if holdingDescriptor is None else self.placeAgents
    observations = []; actions = []; descriptor = None

    for agent in agents:
      o, a, descriptor, v, epsilon = agent.SenseAndAct(
        holdingDescriptor, descriptor, cloud, rlEnv.GetTableHeight(), t, unbias)
      rlEnv.PlotDescriptors([descriptor])
      if self.plotImages: agent.PlotImages(o, a, descriptor)
      observations.append(o)
      actions.append(a)

    return observations, actions, descriptor, epsilon
    
  def SenseAndActMultiple(self, holdingDescriptor, cloud, t, rlEnv, nSamples):
    '''TODO'''
    
    agents = self.graspAgents if holdingDescriptor is None else self.placeAgents
    descriptors = [None] * nSamples; probabilities = ones(nSamples)
    
    # detect grasps/places    
    for agent in agents:
      descriptors, probabilities = agent.SenseAndActMultiple(
        holdingDescriptor, descriptors, probabilities, cloud, rlEnv.GetTableHeight(), t)
    
    return descriptors, probabilities

  def SetInitialDescriptor(self, offset):
    '''TODO'''
    
    self.graspAgents[0].SetInitialDescriptor(offset)
    self.placeAgents[0].SetInitialDescriptor(offset)

  def UpdateQFunctionMonteCarlo(self):
    '''Trains each level of the agent using the Monte Carlo update. Use with AddExperienceMonteCarlo.'''
    
    gc.collect() # necessary because, if low on memory, Tensorflow could crash    
    
    losses = []
    for agent in self.agents:
      loss = agent.UpdateQFunctionMonteCarlo()
      losses.append(loss)

    return losses