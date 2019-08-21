#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly generated cylinders.'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# python
import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from numpy import array, mean
# self
import point_cloud
from rl_environment_pegs_on_disks import RlEnvironmentPegsOnDisks

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  randomSeed = 0
  
  # environment
  removeTable = True
  
  # objects
  objectHeight = [0.01, 0.02]
  objectRadius = [0.03, 0.06]
  nObjects = 1000

  # view
  viewCenter = array([0,0,0])
  viewKeepout = 0.60
  viewWorkspace = [(-1.0,1.0),(-1.0,1.0),(-1.0,1.0)]
  voxelSize = 0.002

  # visualization/saving
  showViewer = False
  showSteps = False
  plotImages = False  

  # INITIALIZATION =================================================================================

  params = locals()
  seed(randomSeed)
  rlEnv = RlEnvironmentPegsOnDisks(params)

  # RUN TEST =======================================================================================

  for objIdx in xrange(nObjects):

    obj = rlEnv.GenerateCylinderMesh(objectHeight, objectRadius, "disk-{}".format(objIdx))
    cloud = rlEnv.GetFullCloud(viewCenter, viewKeepout, viewWorkspace, False, voxelSize)
    
    data = {"cloud":cloud, "height":obj.height, "radius":obj.radius}
    savemat("disk-{}.mat".format(objIdx), data)

    rlEnv.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud)
    if showSteps: raw_input("Placed disk-{}.".format(objIdx))

    rlEnv.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()