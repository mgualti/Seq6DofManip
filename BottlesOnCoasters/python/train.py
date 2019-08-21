#!/usr/bin/env python
'''Trains an HVS agent to perform the bottles on coasters task.'''

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
from rl_environment_bottles_on_coasters import RlEnvironmentBottlesOnCoasters
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
  rlEnv = RlEnvironmentBottlesOnCoasters(params)
  rlAgent = RlAgent(params)

  # if testing, load previous results
  if loadNetwork:
    rlAgent.LoadQFunction()

  if loadDatabase:
    rlAgent.LoadExperienceDatabase()

  # RUN TEST =======================================================================================

  episodeReturn = []; nPlacedObjects = []; nGraspedObjects = []
  episodeTime = []; timeStepEpsilon = []; databaseSize = []; losses = []
  
  for episode in xrange(nEpisodes):

    startTime = time()

    # place random object in random orientation on table
    rlEnv.MoveHandToHoldingPose()
    rlEnv.PlaceObjects(False)
    rlEnv.PlaceObjects(True)
    if showSteps: raw_input("Placed objects.")

    R = 0; nPlaced = 0; nGrasped = 0
    holdingDesc = None; observations = []; actions = []; rewards = []; isGrasp = []
    
    for t in xrange(tMax):
      # get a point cloud
      cloud = rlEnv.GetArtificialCloud()
      #rlEnv.PlotCloud(cloud)
      #if showSteps: raw_input("Acquired cloud.")
      isGrasp.append(holdingDesc is None)
      # get the next action
      o, a, overtDesc, epsilon = rlAgent.SenseAndAct(
        holdingDesc, cloud, t, rlEnv, episode >= unbiasOnEpisode)
      # perform transition
      holdingDesc, r = rlEnv.Transition(overtDesc, cloud)
      # save experiences
      timeStepEpsilon.append(epsilon)
      observations.append(o)
      actions.append(a)
      rewards.append(r)
      R += r
      # compute task success -- number of objects placed
      if isGrasp[-1]:
        if holdingDesc is not None:
          nGrasped += 1
        if r < 0:
          nPlaced -= 1
      else:
        if r == 1:
          nPlaced += 1
          
    rlAgent.AddExperienceMonteCarlo(observations, actions, rewards, isGrasp)

    # cleanup episode
    rlEnv.ResetEpisode()
    print("Episode {} had return {}".format(episode, R))

    # training
    if episode % trainEvery == trainEvery-1:
      losses.append(rlAgent.UpdateQFunctionMonteCarlo())
      rlAgent.SaveQFunction()

    # save results
    episodeReturn.append(R)
    nPlacedObjects.append(nPlaced)
    nGraspedObjects.append(nGrasped)
    episodeTime.append(time()-startTime)
    databaseSize.append(rlAgent.GetNumberOfExperiences())

    if episode % trainEvery == trainEvery-1 or episode == nEpisodes-1:
      saveData = {"episodeReturn":episodeReturn, "nPlacedObjects":nPlacedObjects,
        "nGraspedObjects":nGraspedObjects, "episodeTime":episodeTime,
        "timeStepEpsilon":timeStepEpsilon, "databaseSize":databaseSize, "losses":losses}
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