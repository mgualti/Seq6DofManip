'''Class and utilities for lookahead HSA agent. Manages several levels of type RlAgentLookaheadLevel.'''

# python
import os
# scipy
# tensorflow
# self
from rl_agent_lookahead_level_0 import RlAgentLookaheadLevel0
from rl_agent_lookahead_level_1 import RlAgentLookaheadLevel1
from rl_agent_lookahead_level_2 import RlAgentLookaheadLevel2

# AGENT ============================================================================================

class RlAgentLookahead():

  def __init__(self, params):
    '''Initializes a hierarchical SE(2) sampling (HSE2S) agent.
    - Input params: Hyperparameters dictionary.
    '''

    # parameters

    self.deviceId = params["deviceId"]
    self.gamma = params["gamma"]
    self.nLevels = params["nLevels"]
    self.plotImages = params["plotImages"]

    # internal variables

    self.updateRule = None

    # decide which device to use

    if self.deviceId >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.deviceId)

    # initialize a sense agent at each level

    self.senseAgents = []
    self.senseAgents.append(RlAgentLookaheadLevel0(0, params))
    if self.nLevels > 1:
      self.senseAgents.append(RlAgentLookaheadLevel1(1, params))
    if self.nLevels > 2:
      self.senseAgents.append(RlAgentLookaheadLevel2(2, params))

  def AddExperienceMonteCarlo(self, observations, rewards):
    '''TODO'''

    self.CheckUpdateRule("MonteCarlo")
    tMax = len(observations)

    for t in xrange(tMax):
      
      Return = 0
      for i in xrange(t, tMax):
        Return += self.gamma**(i - t) * rewards[i]
      
      for l in xrange(self.nLevels):

        state = observations[t][l]
        experience = [state, Return]
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

  def SenseAndAct(self, holdingDescriptor, t, rlEnv, unbias):
    '''Encodes the agent's observation and maximizes over actions given that observation.
    - Input holdingDescriptor: A descriptor of the last successful grasp.
    - Input t: The current time step.
    - Input rlEnv: The environment, for image generation and plotting.
    - Input epsilon: The current exploration factor.
    - Input plotImages: If True, plots images illustrating the current observation and action.
    - Returns observations: An observation for each level of the form [I_1, I_2] where I_1 is an
      image describing the hand contents and I_2 is an image describing the hand target.
    - Returns actions: An action for each level in the agent's local coordinate frame.
    - Returns descriptor: A description of the action in the world frame, i.e. the target hand pose.
    '''

    observations = []; descriptor = None

    for agent in self.senseAgents:
      o, descriptor, v, epsilon = agent.SenseAndAct(
        holdingDescriptor, descriptor, t, rlEnv, unbias)
      rlEnv.PlotDescriptors([descriptor])
      if self.plotImages: agent.PlotImages(o)
      observations.append(o)

    return observations, descriptor, epsilon

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