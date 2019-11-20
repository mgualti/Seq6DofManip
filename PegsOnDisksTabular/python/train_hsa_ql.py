#!/usr/bin/env python
'''Trains a hierarcical spatial attention (HSA) agent in the tabular pegs on disks domain.'''

# python
import sys
from time import time
# scipy
from scipy.io import loadmat, savemat
from numpy.random import seed
# self
from rl_environment import RlEnvironment
from rl_agent_hsa_ql import RlAgentHsaQl

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
  saveFileName = params["saveFileName"]
  
  # INITIALIZATION =================================================================================
  
  # set random seeds
  seed(randomSeed)

  # initialize agent
  agent = RlAgentHsaQl(params)
  
  if loadQFunction:
    agent.LoadQFunction()
  
  # RUN TEST =======================================================================================
  
  episodeReturn = []; episodeTime = []

  for episode in xrange(nEpisodes):

    startTime = time()
    episodeReturn.append(0)
    if episode >= unbiasOnEpisode: epsilon = 0
    
    env = RlEnvironment(params)
    s = env.GetState()
    
    a, i, o = agent.GetActionsAndObservations(s, epsilon)

    for t in xrange(tMax):
      r = env.Transition(a)
      ss = env.GetState()
      aa, ii, oo = agent.GetActionsAndObservations(ss, epsilon)
      agent.UpdateQFunction(o, i, r, oo)
      s = ss; a = aa; o = oo; i = ii
      episodeReturn[-1] += r

    print("Episode {}.{} had return {}.".format(realization, episode, episodeReturn[-1]))
    episodeTime.append(time()-startTime)

    print("Agent learned {} values.".format(agent.GetQTableSize()))

  saveData = {"episodeReturn":episodeReturn, "episodeTime":episodeTime,
    "nValuesLearned":agent.GetQTableSize()}
  saveData.update(params)
  savemat(saveFileName, saveData)
  agent.SaveQFunction()

if __name__ == "__main__":
  
  if len(sys.argv) < 2 or type(sys.argv[1]) != type(""):
    print("Usage: trainFile paramsFile")
    exit()
  
  realization = 0 if len(sys.argv) < 3 else int(sys.argv[2])
  exec("from " + sys.argv[1] + " import Parameters")
  
  Parameters(realization)
  Main()
