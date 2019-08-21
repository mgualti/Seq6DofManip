#!/usr/bin/env python
'''Generates mesh files and point clouds for randomly selected bottles from 3DNet.'''

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
  objectDirectory = "/home/mgualti/Data/3DNet/Cat10_ModelDatabase/bottle"
  objectsToSkip = ["8cf1cc180de4ecd0a826d25a0acb0476.ply"] # z-axis not up
  objectsToSkip += ["b0652a09588f293c7e95755f464f6241.ply", \
    "91a1f4e7a5eab4eab05e5be75e72ca3c.ply", "70e77c09aca88d4cf76fe74de9d44699.ply"] # difficult to grasp
  objectScale = [0.10, 0.20]
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
    
    # Load mesh, save mesh, and get points and normals.
    objectName = "bottle-{}".format(objIdx)
    obj = rlEnv.GenerateMeshFromMesh(objectDirectory, objectScale, objectsToSkip, objectName)
    cloud, normals = rlEnv.GetFullCloud(viewCenter, viewKeepout, viewWorkspace,
      add45DegViews=False, computeNormals=True, voxelSize=voxelSize)
    
    # Occasionally, the normals calculation fails. Replace with the nearest normal.
    if isnan(normals).any():
      nanIdx = sum(isnan(normals), axis=1) > 0
      notNanIdx = logical_not(nanIdx)
      nanFreeTree = cKDTree(cloud[notNanIdx, :])
      d, nearestIdx = nanFreeTree.query(cloud[nanIdx, :])
      normals[nanIdx, :] = normals[nearestIdx, :]
      
    # Compute object bounding box
    workspace = array([[min(cloud[:, 0]), max(cloud[:, 0])],
                       [min(cloud[:, 1]), max(cloud[:, 1])],
                       [min(cloud[:, 2]), max(cloud[:, 2])]])
    
    # Save metadata.
    data = {"cloud":cloud, "normals":normals, "workspace":workspace, "scale":obj.scale}
    savemat(objectName + ".mat", data)
    
    # Optional visualization for debugging.
    rlEnv.PlotCloud(cloud)
    if plotImages: point_cloud.Plot(cloud, normals, 5)
    if showSteps: raw_input("Placed " + objectName + ".")
    
    # Remove the object before loading the next one.
    rlEnv.RemoveObjectSet([obj])

if __name__ == "__main__":
  main()