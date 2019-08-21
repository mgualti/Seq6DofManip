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
from numpy import arange, ceil
  
def Parameters(realization):
  '''Specifies simulation hyperparameters.'''
  
  # system
  randomSeed = realization
  
  # problem
  nObjects = 3
  worldSize = 16
  tMax = 2 * nObjects
  rewardShaping = False

  # learning
  nEpisodes = 1000
  unbiasOnEpisode = nEpisodes * 0.95
  epsilon = 0.00
  initQ = arange(tMax, 0, -1) if rewardShaping else ceil((arange(tMax, 0, -1)) / 2.0)
  alpha = 0.02
  gamma = 1.0

  # visualization/saving
  loadQFunction = False
  saveQFunction = False
  plotImages = False
  saveFileName = "results-{}.mat".format(realization)
  
  # save parameter file
  savemat("parameters.mat", locals())