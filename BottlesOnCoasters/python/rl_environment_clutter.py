'''Reinforcement learning (RL) environment for the bottles on coasters domain.'''

# python
import os
import fnmatch
from copy import copy
# scipy
from scipy.io import loadmat
from matplotlib import pyplot
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy.random import choice, rand, randint, randn, uniform
from numpy import arccos, asscalar, argmax, argmin, array, dot, eye, pi, vstack
# openrave
import openravepy
# self
import point_cloud
from hand_descriptor import HandDescriptor
from rl_environment_bottles_on_coasters import RlEnvironmentBottlesOnCoasters

class RlEnvironmentClutter(RlEnvironmentBottlesOnCoasters):

  def __init__(self, params):
    '''Initializes openrave environment, parameters, and first episode.
    - Input params: System parameters data structure.
    '''

    RlEnvironmentBottlesOnCoasters.__init__(self, params)
    
    # parameters
    self.nClutter = self.params["nClutter"]
    self.clutterObjectFolder = self.params["clutterObjectFolder"]
    
    # internal state
    self.clutterObjectFileNames = os.listdir(self.clutterObjectFolder)
    self.clutterObjectFileNames = fnmatch.filter(self.clutterObjectFileNames, "*.dae")
    self.clutter = []
    
  def GetArtificialCloud(self):
    '''Concatenates point cloud data from all objects and support objects.
    - Returns cloud: Point cloud in the base/world reference frame.
    '''
    
    clouds = []
    objects = self.clutter + self.supportObjects + self.objects
    
    for obj in objects:
      cloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      clouds.append(cloud)

    return vstack(clouds)
    
  def PlaceObjects(self, maxPlaceAttempts = 5, workspace = ((-0.18, 0.18), (-0.18, 0.18))):
    '''TODO'''
    
    # reset episode and and place usual objects first
    self.ResetEpisode()
    super(RlEnvironmentClutter, self).PlaceObjects(False)
    super(RlEnvironmentClutter, self).PlaceObjects(True)
    
    # some preparation
    nObjects = self.nClutter
    folderName = self.clutterObjectFolder
    fileNames = self.clutterObjectFileNames
    
    # select file(s)
    fileIdxs = choice(len(fileNames), size=nObjects, replace=False)
    
    # add objects
    for i in xrange(nObjects):
      
      # choose a random object from the folder
      objectName = fileNames[fileIdxs[i]]      
      
      # load object
      if objectName in self.kinBodyCache:
        body = self.kinBodyCache[objectName]
        self.env.AddKinBody(body)
      else:
        # load mesh
        self.env.Load(folderName + "/" + objectName)
        shortObjectName = objectName[:-4]
        body = self.env.GetKinBody(shortObjectName)
        # load points and normals
        data = loadmat(folderName + "/" + shortObjectName + ".mat")
        body.cloud = data["cloud"]
        body.workspace = data["workspace"]
        # add to cache
        self.kinBodyCache[objectName] = body
      
      # select pose for object
      inCollision = True
      for j in xrange(maxPlaceAttempts):

        theta = uniform(0, 2 * pi)
        R = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], theta)
        
        # choose xy position
        xy = array([ \
          uniform(workspace[0][0], workspace[0][1]),
          uniform(workspace[1][0], workspace[1][1])])
        
        # set height
        z = asscalar(body.workspace[2][1]) + self.GetTableHeight() + 0.001
        
        # set transform
        T = eye(4)
        T[0:2, 3] = xy
        T[2, 3] = z
        T = dot(T, R)
        body.SetTransform(T)
        
        if not self.env.CheckCollision(body):
          inCollision = False
          break
        
      if inCollision:
        self.env.Remove(body)
      else:
        self.clutter.append(body)

  def ResetEpisode(self):
    '''Resets all internal variables pertaining to a particular episode, including objects placed.'''
    
    super(RlEnvironmentClutter, self).ResetEpisode()
    if hasattr(self, "clutter"):
      self.RemoveObjectSet(self.clutter)
    self.clutter = []