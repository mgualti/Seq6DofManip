#!/usr/bin/env python
'''Parameters for a temporal difference (TD) agent in the tabular pegs on disks domain.'''

# python
# scipy
from scipy.io import savemat
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
  nEpisodes = 1000000
  initQ = arange(tMax, 0, -1) if rewardShaping else ceil((arange(tMax, 0, -1)) / 2.0)
  alpha = 0.02
  gamma = 1.00

  # visualization/saving
  saveFileName = "results-{}.mat".format(realization)
  
  # save parameter file
  savemat("parameters.mat", locals())