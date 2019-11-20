#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly generated coasters (flat cylinders).'''

# python
import time
# scipy
from scipy.io import savemat
from numpy.random import seed
from numpy import array, max, min
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
  objectExtents = [0.01, 0.04]
  nObjects = 1000

  # view
  viewCenter = array([0, 0, 0])
  viewKeepout = 0.60
  viewWorkspace = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
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
    objectName = "block-{}".format(objIdx)
    obj = rlEnv.GenerateBoxMesh(objectExtents[0], objectExtents[1], objectName)
    cloud = rlEnv.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews=False, computeNormals=False, voxelSize=voxelSize)
      
    # Compute object bounding box
    workspace = array([[min(cloud[:, 0]), max(cloud[:, 0])],
                       [min(cloud[:, 1]), max(cloud[:, 1])],
                       [min(cloud[:, 2]), max(cloud[:, 2])]])

    # Save metadata.
    data = {"cloud":cloud, "workspace":workspace}
    savemat(objectName + ".mat", data)
    
    # Optional visualization for debugging.
    rlEnv.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud)
    if showSteps: raw_input("Placed " + objectName + ".")
    
    # Remove the object before loading the next one.
    rlEnv.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()