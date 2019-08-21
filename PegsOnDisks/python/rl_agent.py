'''Class and utilities for an HVS agent.'''

# python
import os
import shutil
from copy import copy
from time import time
# scipy
from matplotlib import pyplot
from numpy.random import choice, permutation, rand, randint
from numpy import argmax, arange, array, cumsum, exp, fliplr, flipud, logical_not, isinf, mean, \
  pi, reshape, stack, where, zeros
# tensorflow
import tensorflow
from tensorflow import keras
# self
from rl_agent_level_0 import RlAgentLevel0
from rl_agent_level_1 import RlAgentLevel1
from rl_agent_level_2 import RlAgentLevel2
from rl_agent_level_3 import RlAgentLevel3

# AGENT ============================================================================================

class RlAgent():

  def __init__(self, params):
    '''Initializes a hierarchical volume sampling (HVS) agent.
    - Input params: Hyperparameters dictionary.
    '''

    # parameters

    self.deviceId = params["deviceId"]
    self.gamma = params["gamma"]
    self.plotImages = params["plotImages"]

    # internal variables

    self.updateRule = None

    # decide which device to use

    if self.deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.deviceId)

    # initialize a sense agent at each level

    self.senseAgents = []
    self.senseAgents.append(RlAgentLevel0(0, params))
    self.senseAgents.append(RlAgentLevel1(1, params))
    self.senseAgents.append(RlAgentLevel2(2, params))
    self.senseAgents.append(RlAgentLevel3(3, params))
    self.nLevels = len(self.senseAgents)

    # set up logging

    self.tfLogDir = os.getcwd() + "/tensorflow/logs"
    if os.path.exists(self.tfLogDir):
      shutil.rmtree(self.tfLogDir)

    tensorflow.summary.FileWriter(self.tfLogDir, keras.backend.get_session().graph)

  def AddExperienceMonteCarlo(self, observations, actions, rewards):
    '''Adds an experience, everything needed for a Monte Carlo update, to the agent's replay database.
    - Input observations: List of lists, [[o_1, ..., o_l]_0, ..., [o_1, ..., o_l]_T], an observation
      at each time step and level.
    - Input actions: An action at each time step and level.
    - Input rewards: A reward, one for each time step (from t=0 to t=T).
    - Returns None.
    '''

    self.CheckUpdateRule("MonteCarlo")
    tMax = len(observations)

    for t in xrange(tMax):
      
      Return = 0
      for i in xrange(t, tMax):
        Return += self.gamma**(i - t) * rewards[i]
      
      for l in xrange(self.nLevels):

        state = observations[t][l]
        action = actions[t][l]

        experience = [state, action, Return]
        self.senseAgents[l].AddExperience(experience)
        
  def AddExperienceQLearning(self, o, a, r, oo):
    '''nLevels-step Q-learning.'''

    self.CheckUpdateRule("QLearning")

    for l in xrange(self.nLevels):

      experience = [o[l], a[l], r, oo[l]]
      self.senseAgents[l].AddExperience(experience)
        
  def AddExperienceSarsa(self, o, a, r, oo, aa):
    '''nLevels-step Sarsa.'''

    self.CheckUpdateRule("Sarsa")

    for l in xrange(self.nLevels):

      experience = [o[l], a[l], r, oo[l], aa[l]]
      self.senseAgents[l].AddExperience(experience)

  def CheckUpdateRule(self, name):
    '''Throws an error if the update rule is not "name".'''

    if self.updateRule:
      if self.updateRule != name:
        raise Exception("Currently using the update rule {}, but attempted an {} update!".format(\
          self.updateRule, name))
    else:
      self.updateRule = name

  def GetNumberOfExperiences(self):
    '''Returns the number of experiences in the experience replay database at each level.'''
    
    nExperiences = 0    
    for i in xrange(self.nLevels):
      nExperiences += self.senseAgents[i].GetNumberOfExperiences()
    return nExperiences

  def LoadExperienceDatabase(self):
    '''Loads an experience replay database for each agent from file into memory.'''

    for agent in self.senseAgents:
      agent.LoadExperienceDatabase()

  def LoadQFunction(self):
    '''Loads a tensorflow model and network weights from file into memory.'''

    for agent in self.senseAgents:
      agent.LoadQFunction()

  def SenseAndAct(self, holdingDescriptor, cloud, t, rlEnv, unbias):
    '''TODO'''

    observations = []; actions = []; descriptor = None

    for agent in self.senseAgents:
      o, a, descriptor, v, epsilon = agent.SenseAndAct(
        holdingDescriptor, descriptor, cloud, rlEnv.GetTableHeight(), t, unbias)
      rlEnv.PlotDescriptors([descriptor])
      if self.plotImages: agent.PlotImages(o, a, descriptor)
      observations.append(o)
      actions.append(a)

    return observations, actions, descriptor, epsilon

  def SaveExperienceDatabase(self):
    '''Saves the experience replay databases to file.'''

    for agent in self.senseAgents:
      agent.SaveExperienceDatabase()

  def SaveQFunction(self):
    '''Saves the tensorflow models and weights to file.'''

    for agent in self.senseAgents:
      agent.SaveQFunction()

  def UpdateQFunctionMonteCarlo(self):
    '''Trains each level of the agent using the Monte Carlo update. Use with AddExperienceMonteCarlo.'''

    self.CheckUpdateRule("MonteCarlo")

    losses = []
    for agent in self.senseAgents:
      loss = agent.UpdateQFunctionMonteCarlo()
      losses.append(loss)

    return losses
    
  def UpdateQFunctionQLearning(self):
    '''Update for nLevels-step Q-learning.'''

    self.CheckUpdateRule("QLearning")

    losses = []
    for agent in self.senseAgents:
      loss = agent.UpdateQFunctionQLearning()
      losses.append(loss)

    return losses
    
  def UpdateQFunctionSarsa(self):
    '''Update for nLevels-step Sarsa.'''

    self.CheckUpdateRule("Sarsa")

    losses = []
    for agent in self.senseAgents:
      loss = agent.UpdateQFunctionSarsa()
      losses.append(loss)

    return losses