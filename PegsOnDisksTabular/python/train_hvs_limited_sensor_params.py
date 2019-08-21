#!/usr/bin/env python
'''TODO'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# python
import os
# scipy
from scipy.io import loadmat, savemat
from numpy.random import choice, normal, uniform
  
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
  nEpisodes = 500000
  unbiasOnEpisode = nEpisodes * 0.95
  epsilon = 0.00
  initQ = [4, 3, 2, 1] if rewardShaping else [3, 3, 2, 2, 1, 1]
  alpha = 0.02
  gamma = 1.0

  # visualization/saving
  loadQFunction = False
  plotImages = False
  saveFileName = "results-{}.mat".format(realization)
  
  # save parameter file
  savemat("parameters.mat", locals())