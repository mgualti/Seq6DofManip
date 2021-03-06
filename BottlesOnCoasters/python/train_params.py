#!/usr/bin/env python
'''Default parameters for an HSA agent in the bottles on coasters domain.'''

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
  nEpisodes = 200000
  trainEvery = 1000
  maxExperiences = 50000
  epsilonMin = 0.03
  unbiasOnEpisode = 0.95 * nEpisodes
  
  # misc
  gamma = 0.00
  includeTimeGrasp = True
  includeTimePlace = False
  nGraspOrientations = 60
  nPlaceOrientations = 3

  # hand descriptor
  imP = 48
  imD = [0.2375, 0.4750, 0.4750, 0.4750, 0.4750] # maxObjHeight + handDepth / 2
  imW = [0.3600, 0.1050, 0.1050, 0.0900, 0.3000]
  selD = [0.1375, 0.09, 0.0225]
  selW = [0.3600, 0.09, 0.0225]

  # network parametrs
  conv1KernelSize = 8
  conv1Outputs = 64
  conv1Stride = 2
  conv2KernelSize = 4
  conv2Outputs = 64
  conv2Stride = 2
  conv3KernelSize = 3
  conv3Outputs = 64
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
  objectFolder = "/home/mgualti/Data/Seq6DofManip/Bottles"
  supportObjectFolder = "/home/mgualti/Data/Seq6DofManip/Coasters"
  graspFrictionCone = 15 # degrees
  graspMinDistFromBottom = 0.04 # meters
  placeOrientTolerance = 3.0 # degrees
  placePosTolerance = 0.02 # meters
  placeHeightTolerance = [0.02, 0.02] # meters
  
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