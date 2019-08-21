#!/usr/bin/env python
'''Parameters for an Monte Carlo HSE2S agent for the pegs on disks task.'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# python
import os
# scipy
from scipy.io import loadmat, savemat
from numpy.random import choice, normal, uniform
from numpy import array
  
def Parameters(realization):
  '''Specifies simulation hyperparameters.'''
  
  # === AGENT ===
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "Seq6DofManip2" in cwd else (0 if "Seq6DofManip1" in cwd else -1)
  randomSeed = realization
  modelFolder = "pegs-on-disks"
  
  # curriculum
  tMax = 4
  nEpisodes = 100000
  trainEvery = 1000
  maxExperiences = 50000
  epsilonMin = 1.0 - (1.0 - 0.04)**3
  unbiasOnEpisode = 0.95 * nEpisodes
  
  # misc
  gamma = 0.00
  nLevels = 1

  # hand descriptor
  imP = 64
  imD = 0.2375 # 2 * maxObjectHeight + halfHandDepth
  imW = 0.09

  # network parametrs (per level)
  conv1KernelSize = [8, 8, 8]
  conv1Outputs = [32, 32, 32]
  conv1Stride = [1, 2, 2]
  conv2KernelSize = [4, 4, 4]
  conv2Outputs = [64, 64, 64]
  conv2Stride = [1, 2, 2]
  conv3KernelSize = [3, 3, 3]
  conv3Outputs = [32, 32, 32]
  conv3Stride = [1, 2, 2]
  conv4KernelSize = [2, 2, 2]
  conv4Outputs = [64, 4, 4]
  conv4Stride = [1, 2, 2]

  # optimization parametrs
  nEpochs = 1
  batchSize = 64
  weightDecay = 0.0001
  baseLearningRate = 0.0001
  optimizer = "Adam"  

  # === ENVIRONMENT ===
  
  # objects
  nObjects = 2
  nSurfaceObjects = 2
  objHeight = [0.03, 0.10]
  objRadius = [0.01, 0.02]
  surfObjHeight = [0.01, 0.02]
  surfObjRadius = [0.03, 0.06]
  placeHeightTolerance = [0.01, 0.02]
  
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
    plotImages = True
  
  # save parameter file
  savemat("parameters.mat", locals())