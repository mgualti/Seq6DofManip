#!/usr/bin/env python
'''Parameters for an HSA agent with 3 levels.'''

# python
import os
# scipy
from scipy.io import savemat
  
def Parameters(realization):
  '''Specifies simulation hyperparameters.'''
  
  # === AGENT ===
  
  # system
  cwd = os.getcwd()
  deviceId = 1 if "Seq6DofManip2" in cwd else (0 if "Seq6DofManip1" in cwd else -1)
  randomSeed = realization
  modelFolder = "models"
  
  # curriculum
  tMax = 4
  nEpisodes = 80000
  trainEvery = 1000
  maxExperiences = 50000
  epsilonMin = 0.04
  unbiasOnEpisode = nEpisodes - 5000
  
  # misc
  gamma = 0.00
  nLevels = 3

  # hand descriptor
  imP =  64
  imD =  [0.1375, 0.2375, 0.2375, 0.2375] # 2 * maxObjectHeight + halfHandDepth
  imW =  [0.3600, 0.0900, 0.0900, 0.0900]
  selD = [0.1375, 0.0900, 0.0225]
  selW = [0.3600, 0.0900, 0.0225]

  # network parametrs (per level)
  conv1KernelSize = [ 7,  7,  7]
  conv1Outputs =    [32, 32, 32]
  conv1Stride =     [ 2,  2,  2]
  conv2KernelSize = [ 7,  7,  7]
  conv2Outputs =    [64, 64, 64]
  conv2Stride =     [ 2,  2,  2]
  conv3KernelSize = [ 7,  7,  7]
  conv3Outputs =    [32, 32, 32]
  conv3Stride =     [ 2,  2,  2]
  conv4KernelSize = [ 7,  7,  7]
  conv4Outputs =    [32, 32, 32]
  conv4Stride =     [ 2,  2,  2]
  conv5KernelSize = [ 7,  7,  7]
  conv5Outputs =    [ 4,  4,  4]
  conv5Stride =     [ 1,  1,  1]

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
