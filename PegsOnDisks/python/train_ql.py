#!/usr/bin/env python
'''Trains a Monte Carlo HSE2S agent to perform the pegs on disks task.'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# uncomment when profiling
#import os; os.chdir("/home/mgualti/Seq6DofManip/PickAndPlace")

# python
import sys
from time import time
# scipy
from scipy.io import loadmat, savemat
from numpy.random import seed
# tensorflow
import tensorflow
# self
from rl_environment_pegs_on_disks import RlEnvironmentPegsOnDisks
from rl_agent import RlAgent

def Main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================

  params = loadmat("parameters.mat", squeeze_me=True)
  randomSeed = params["randomSeed"]
  tMax = params["tMax"]
  nEpisodes = params["nEpisodes"]
  trainEvery = params["trainEvery"]
  unbiasOnEpisode = params["unbiasOnEpisode"]
  saveFileName = params["saveFileName"]
  loadNetwork = params["loadNetwork"]
  loadDatabase = params["loadDatabase"]
  showSteps = params["showSteps"]

  # INITIALIZATION =================================================================================

  # set random seeds
  seed(randomSeed)
  tensorflow.set_random_seed(randomSeed)

  # initialize agent and environment
  rlEnv = RlEnvironmentPegsOnDisks(params)
  rlAgent = RlAgent(params)

  # if testing, load previous results
  if loadNetwork:
    rlAgent.LoadQFunction()

  if loadDatabase:
    rlAgent.LoadExperienceDatabase()

  # RUN TEST =======================================================================================

  episodeReturn = []; nPlacedObjects = []; episodeTime = []; timeStepEpsilon = [];
  databaseSize = []; losses = []
  
  for episode in xrange(nEpisodes):

    startTime = time()

    # place random object in random orientation on table
    rlEnv.MoveHandToHoldingPose()
    rlEnv.PlaceObjects(False)
    rlEnv.PlaceObjects(True)
    if showSteps: raw_input("Placed objects.")

    R = 0; nPlaced = 0; holdingDesc = None; o = None; a = None; r = None
    for t in xrange(tMax):
      # get a point cloud
      cloud = rlEnv.GetArtificialCloud()
      isGrasp = holdingDesc is None
      # get the next action
      oo, aa, overtDesc, epsilon = rlAgent.SenseAndAct(
        holdingDesc, cloud, t, rlEnv, episode >= unbiasOnEpisode)
      # perform transition
      holdingDesc, rr = rlEnv.Transition(overtDesc, cloud)
      # save experience
      if t > 0: rlAgent.AddExperienceQLearning(o, a, r, oo)
      o = oo; a = aa; r = rr
      # save recorded data
      R += r
      if isGrasp:
        if r < 0:
          nPlaced -= 1
      else:
        if r > 0:
          nPlaced += 1
      timeStepEpsilon.append(epsilon)
    
    # add final experience
    rlAgent.AddExperienceQLearning(o, a, r, [None]*rlAgent.nLevels)

    # cleanup episode
    rlEnv.ResetEpisode()
    print("Episode {} had return {}".format(episode, R))

    # training
    if episode % trainEvery == trainEvery-1:
      losses.append(rlAgent.UpdateQFunctionQLearning())
      rlAgent.SaveQFunction()

    # save results
    episodeReturn.append(R)
    nPlacedObjects.append(nPlaced)
    episodeTime.append(time()-startTime)
    databaseSize.append(rlAgent.GetNumberOfExperiences())

    if episode % trainEvery == trainEvery-1 or episode == nEpisodes-1:
      saveData = {"episodeReturn":episodeReturn, "nPlacedObjects":nPlacedObjects,
        "episodeTime":episodeTime, "timeStepEpsilon":timeStepEpsilon,
        "databaseSize":databaseSize, "losses":losses}
      saveData.update(params)
      savemat(saveFileName, saveData)

    # backup agent data
    if episode == nEpisodes-1:
      rlAgent.SaveExperienceDatabase()

if __name__ == "__main__":
  
  if len(sys.argv) < 2 or type(sys.argv[1]) != type(""):
    print("Usage: trainFile paramsFile")
    exit()
  
  realization = 0 if len(sys.argv) < 3 else int(sys.argv[2])
  exec("from " + sys.argv[1] + " import Parameters")
  
  Parameters(realization)
  Main()