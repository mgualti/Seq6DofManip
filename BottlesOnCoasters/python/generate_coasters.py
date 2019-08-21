#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly generated coasters (flat cylinders).'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# python
import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from scipy.spatial import cKDTree
from numpy import array, isnan, logical_not, logical_or, max, mean, min, sum, where
# self
import point_cloud
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.'''

  # PARAMETERS =====================================================================================
  
  # system
  randomSeed = 0
  
  # environment
  removeTable = True
  
  # objects
  objectHeight = [0.005, 0.020]
  objectRadius = [0.040, 0.060]
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
  rlEnv = RlEnvironment(params)

  # RUN TEST =======================================================================================

  for objIdx in xrange(nObjects):
    
    # Generate mesh, save mesh, and get points.
    objectName = "coaster-{}".format(objIdx)
    obj = rlEnv.GenerateCylinderMesh(objectHeight, objectRadius, objectName)
    cloud = rlEnv.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews=False, computeNormals=False, voxelSize=voxelSize)
      
    # Compute object bounding box
    workspace = array([[min(cloud[:, 0]), max(cloud[:, 0])],
                       [min(cloud[:, 1]), max(cloud[:, 1])],
                       [min(cloud[:, 2]), max(cloud[:, 2])]])

    # Save metadata.
    data = {"cloud":cloud, "workspace":workspace, "height":obj.height, "radius":obj.radius}
    savemat(objectName + ".mat", data)
    
    # Optional visualization for debugging.
    rlEnv.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud)
    if showSteps: raw_input("Placed " + objectName + ".")
    
    # Remove the object before loading the next one.
    rlEnv.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()