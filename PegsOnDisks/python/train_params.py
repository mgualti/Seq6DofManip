#!/usr/bin/env python
'''Parameters for an Monte Carlo HSA agent for the pegs on disks task.'''

# python
import os
# scipy
from scipy.io import loadmat, savemat
from numpy.random import choice, normal, uniform
from numpy import cos, pi
  
def Parameters(realization):
  '''Specifies simulation hyperparameters.'''
  
  # === AGENT ===
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "Seq6DofManip2" in cwd else (0 if "Seq6DofManip1" in cwd else -1)
  randomSeed = realization
  
  # curriculum
  tMax = 4
  nEpisodes = 100000
  trainEvery = 1000
  maxExperiences = 50000
  epsilonMin = 0.03
  unbiasOnEpisode = 0.95 * nEpisodes
  
  # misc
  gamma = 0.25
  nGraspOrientations = 60
  nPlaceOrientations = 3

  # hand descriptor
  imP = 48
  imD = [0.1375, 0.2750, 0.2750, 0.2750, 0.2750] # maxObjHeight + handDepth / 2
  imW = [0.3600, 0.0900, 0.0900, 0.0900, 0.1000]
  selD = [0.1375, 0.09, 0.0225]
  selW = [0.3600, 0.09, 0.0225]

  # network parametrs
  conv1KernelSize = 8
  conv1Outputs = 32
  conv1Stride = 2
  conv2KernelSize = 4
  conv2Outputs = 64
  conv2Stride = 2
  conv3KernelSize = 3
  conv3Outputs = 32
  conv3Stride = 2
  conv4KernelSize = 2
  conv4Outputs = 6
  conv4Stride = 1

  # optimization parametrs
  nEpochs = 1
  batchSize = 64
  weightDecay = 0.0000
  baseLearningRate = 0.0001
  optimizer = "Adam"  

  # === ENVIRONMENT ===
  
  # objects
  nObjects = 2
  nSupportObjects = 2
  objectFolder = "/home/mgualti/Data/Seq6DofManip/Pegs"
  supportObjectFolder = "/home/mgualti/Data/Seq6DofManip/Disks"
  placeOrientTolerance = 1 - cos(1.0 * pi / 180.0)
  placeHeightTolerance = [0.02, 0.02]
  rewardCapGrasps = True
  
  # misc
  removeTable = False

  # === Visualization / Saving ===

  saveFileName = "results-{}.mat".format(realization)
  loadNetwork = False
  loadDatabase = False
  showViewer = False
  showSteps = False
  plotImages = False

  # visualize policy
  visualizePolicy = False
  if visualizePolicy:
    randomSeed = randomSeed + 1
    trainEvery = nEpisodes + 1
    unbiasOnEpisode = 0
    loadNetwork = True
    loadDatabase = False
    showViewer = True
    showSteps = True
    plotImages = False
  
  # save parameter file
  savemat("parameters.mat", locals())