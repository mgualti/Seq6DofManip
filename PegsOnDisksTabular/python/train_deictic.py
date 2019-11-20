#!/usr/bin/env python
'''Trains a Deictic Image Mapping agent in the tabular pegs on disks domain.'''

# python
import sys
from time import time
# scipy
from scipy.io import loadmat, savemat
from numpy.random import seed
# self
from rl_environment import RlEnvironment
from rl_agent_deictic import RlAgentDeictic

def Main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  params = loadmat("parameters.mat", squeeze_me=True)
  randomSeed = params["randomSeed"]
  tMax = params["tMax"]
  nEpisodes = params["nEpisodes"]
  unbiasOnEpisode = params["unbiasOnEpisode"]
  epsilon = params["epsilon"]
  loadQFunction = params["loadQFunction"]
  saveQFunction = params["saveQFunction"]
  saveFileName = params["saveFileName"]
  
  # INITIALIZATION =================================================================================
  
  # set random seeds
  seed(randomSeed)

  # initialize agent
  agent = RlAgentDeictic(params)
  
  if loadQFunction:
    agent.LoadQFunction()
  
  # RUN TEST =======================================================================================
  
  nPlacedObjects = []; episodeTime = []

  for episode in xrange(nEpisodes):

    startTime = time()
    R = 0
    nPlacedObjects.append(0)
    if episode >= unbiasOnEpisode: epsilon = 0
    
    env = RlEnvironment(params)
    s = env.GetState()
    
    a, i, o = agent.GetActionAndObservation(s, epsilon)

    for t in xrange(tMax):
      isPlace = env.IsPlace()
      r = env.Transition(a)
      if isPlace: nPlacedObjects[-1] += r > 0
      R += r
      ss = env.GetState()
      aa, ii, oo = agent.GetActionAndObservation(ss, epsilon)
      agent.UpdateQFunction(o, i, r, oo, ii)
      s = ss; a = aa; o = oo; i = ii

    print("Episode {}.{} had return {}.".format(realization, episode, R))
    episodeTime.append(time() - startTime)

    print("Agent learned {} values.".format(agent.GetQTableSize()))

  saveData = {"nPlacedObjects":nPlacedObjects, "episodeTime":episodeTime, 
    "nValuesLearned":agent.GetQTableSize()}
  saveData.update(params)
  savemat(saveFileName, saveData)
  if saveQFunction: agent.SaveQFunction()

if __name__ == "__main__":
  
  if len(sys.argv) < 2 or type(sys.argv[1]) != type(""):
    print("Usage: trainFile paramsFile")
    exit()
  
  realization = 0 if len(sys.argv) < 3 else int(sys.argv[2])
  exec("from " + sys.argv[1] + " import Parameters")
  
  Parameters(realization)
  Main()
