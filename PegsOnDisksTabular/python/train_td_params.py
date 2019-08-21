#!/usr/bin/env python
'''Parameters for a temporal difference (TD) agent in the tabular pegs on disks domain.'''

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
  nObjects = 2
  worldSize = 2
  tMax = 2 * nObjects
  rewardShaping = False

  # learning
  nEpisodes = 1000000
  initQ = arange(tMax, 0, -1) if rewardShaping else ceil((arange(tMax, 0, -1)) / 2.0)
  alpha = 0.02
  gamma = 1.00

  # visualization/saving
  saveFileName = "results-{}.mat".format(realization)
  
  # save parameter file
  savemat("parameters.mat", locals())