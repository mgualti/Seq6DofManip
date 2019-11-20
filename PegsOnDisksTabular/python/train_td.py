#!/usr/bin/env python
'''Trains a temporal difference (TD) agent in the tabular pegs on disks domain.'''

# python
import sys
from time import time
# scipy
from scipy.io import loadmat, savemat
from numpy.random import seed
# self
from rl_environment import RlEnvironment
from rl_agent_td import RlAgentTd

def Main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  params = loadmat("parameters.mat", squeeze_me=True)
  randomSeed = params["randomSeed"]
  tMax = params["tMax"]
  nEpisodes = params["nEpisodes"]
  saveFileName = params["saveFileName"]
  
  # INITIALIZATION =================================================================================
  
  # set random seeds
  seed(randomSeed)

  # initialize agent
  agent = RlAgentTd(params)
  
  # RUN TEST =======================================================================================
  
  episodeReturn = []; episodeTime = []

  for episode in xrange(nEpisodes):

    startTime = time()
    episodeReturn.append(0)

    env = RlEnvironment(params)
    s = env.GetState()
    a = agent.GetAction(s)

    for t in xrange(tMax):
      r = env.Transition(a)
      ss = env.GetState()
      aa = agent.GetAction(ss)
      agent.UpdateQFunction(s, a, r, ss, aa)
      s = ss; a = aa
      episodeReturn[-1] += r

    print("Episode {}.{} had return {}.".format(realization, episode, episodeReturn[-1]))
    episodeTime.append(time()-startTime)

    print("Agent learned {} values.".format(agent.GetQTableSize()))

  saveData = {"episodeReturn":episodeReturn, "episodeTime":episodeTime,
    "nValuesLearned":agent.GetQTableSize()}
  saveData.update(params)
  savemat(saveFileName, saveData)

if __name__ == "__main__":
  
  if len(sys.argv) < 2 or type(sys.argv[1]) != type(""):
    print("Usage: trainFile paramsFile")
    exit()
  
  realization = 0 if len(sys.argv) < 3 else int(sys.argv[2])
  exec("from " + sys.argv[1] + " import Parameters")
  
  Parameters(realization)
  Main()
