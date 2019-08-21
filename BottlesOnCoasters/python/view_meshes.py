#!/usr/bin/env python
'''Script for viewing mesh files with the OpenRAVE viewer.'''

# python
import os
import fnmatch
# scipy
# self
from rl_environment import RlEnvironment

def main():
  '''Entrypoint to the program.'''
  # PARAMETERS =====================================================================================
  
  # objects
  meshDirectory = "/home/mgualti/Data/3DNet/Cat10_ModelDatabase/bottle/"
  objectsToSkip = ["8cf1cc180de4ecd0a826d25a0acb0476.ply"] # z-axis not up
  objectsToSkip += ["b0652a09588f293c7e95755f464f6241.ply", \
    "91a1f4e7a5eab4eab05e5be75e72ca3c.ply", "70e77c09aca88d4cf76fe74de9d44699.ply"] # difficult to grasp
  meshExtension = "ply"
  scale = 0.25
  
  # environment
  removeTable = True
  showViewer = True

  # INITIALIZATION =================================================================================

  params = locals()
  rlEnv = RlEnvironment(params)

  # RUN TEST =======================================================================================

  fileNames = os.listdir(meshDirectory)
  fileNames = fnmatch.filter(fileNames, "*." + meshExtension)

  for fileName in fileNames:
    
    if fileName in objectsToSkip: continue
    objectName = fileName[:-4]
    rlEnv.env.Load(meshDirectory + "/" + fileName, {'scalegeometry':str(scale)})
    body = rlEnv.env.GetKinBody(objectName)
    raw_input(objectName)
    rlEnv.env.Remove(body)

if __name__ == "__main__":
  main()