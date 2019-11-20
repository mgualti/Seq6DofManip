'''Reinforcement learning (RL) environment for the upright pegs on disks domain.'''

# python
import os
import fnmatch
from copy import copy
from time import sleep, time
# scipy
from scipy.io import loadmat
from matplotlib import pyplot
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy.random import choice, rand, randint, randn, uniform
from numpy import arccos, argmin, array, arange, cos, dot, eye, hstack, mean, pi, power, repeat, \
  reshape, sin, sqrt, sum, vstack, zeros
# openrave
import openravepy
# self
from rl_environment import RlEnvironment

class RlEnvironmentPegsOnDisks(RlEnvironment):

  def __init__(self, params):
    '''Initializes openrave environment, parameters, and episode.
    - Input showViewer: If True, shows the openrave viewer.
    - Input removeTable: If True, the table is not included in the scene.
    '''

    RlEnvironment.__init__(self, params)

    # parameters

    self.colors = array([ \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ])

    # internal state

    self.objects = []
    self.supportObjects = []
    self.ResetEpisode()

  def IsPegGrasp(self, descriptor):
    '''Checks if, when the hand is placed at the descriptor's pose and is closed, if a grasp takes place.
    A grasp must be (1) collision-free and (2) center of top of exactly 1 cylinder must be in hand closing region.
    - Input descriptor: HandDescriptor object of the target hand pose.
    - Returns graspedObject: The handle of the grasped object if a cylinder can be grasped from the
      target hand pose; otherwise None.
    '''

    # Grasp must be collision-free.
    if self.IsRobotInCollision(descriptor):
      return None

    # Top point of exactly 1 cylinder must lie within hand.
    graspedObject = None
    for obj in self.objects:
      topPoint = obj.GetTransform()[0:3, 3]
      topPoint[2] += obj.height / 2.0
      if topPoint[0] > descriptor.center[0] + descriptor.height / 2.0: continue
      if topPoint[0] < descriptor.center[0] - descriptor.height / 2.0: continue
      if topPoint[1] > descriptor.center[1] + descriptor.width / 2.0: continue
      if topPoint[1] < descriptor.center[1] - descriptor.width / 2.0: continue
      if topPoint[2] > descriptor.center[2] + descriptor.depth / 2.0: continue
      if topPoint[2] < descriptor.center[2] - descriptor.depth / 2.0: continue
      if graspedObject is not None:
        # multi-object grasp
        return None
      graspedObject = obj

    return graspedObject

  def IsRobotInCollision(self, descriptor):
    '''Checks collision between the robot and the world. Required due to an ODE bug.
    - Input descriptor: HandDescriptor object for the current hand pose.
    - Returns: True if in collision and False otherwise.
    '''

    self.robot.SetTransform(descriptor.T)
    if self.env.CheckCollision(self.robot):
      return True

    # ODE misses collisions where the finger tip is inside the cylinder from the top.
    # But it is fairly simple to check for this...

    hh = 0.5 * descriptor.height
    bUp = descriptor.top + hh * descriptor.axis
    bDn = descriptor.top - hh * descriptor.axis
    v1 = 0.5 * descriptor.width * descriptor.binormal
    v2 = (0.5 * descriptor.width + 0.01) * descriptor.binormal
    checkPoints = [bUp + v1, bUp + v2, bDn + v1, bDn + v2, bUp - v1, bUp - v2, bDn - v1, bDn - v2]

    objects = self.objects + self.supportObjects
    for obj in objects:
      c = obj.GetTransform()[0:3, 3]
      rr = obj.radius**2
      hh = obj.height / 2.0
      for p in checkPoints:
        if p[2] > c[2] - hh and p[2] < c[2] + hh and sum(pow(p[0:2] - c[0:2], 2)) < rr:
          return True

    return False
    
  def PerformGrasp(self, descriptor):
    '''Simulates a grasp by grasping the held object and moving it to a holding pose.
    Assumes IsPegGrasp is true and self.holdingObject is set to the grasped object.
    - Input descriptor: Pose of the grasp.
    - Returns reward: -1 if grasping a placed object and 1 otherwise.
    '''
    
    # generate grasp image
    descriptor.GenerateHeightmap(self)
    self.holdingDescriptor = descriptor
    
    # center object in hand
    bTo = self.holdingObject.GetTransform()
    hTb = inv(descriptor.T)
    hTo = dot(hTb, bTo)
    hTo[1, 3] = 0
    bToNew = dot(descriptor.T, hTo)
    self.holdingObject.SetTransform(bToNew)
    
    # move to holding pose
    self.MoveHandToHoldingPose()
    self.MoveObjectToHandAtGrasp(descriptor.T, self.holdingObject)
    
    # compute reward
    if self.holdingObject in self.placedObjects:
      del self.placedObjects[self.holdingObject]
      return -1.0
    return 1.0
    
  def PerformPlace(self, descriptor):
    '''Places the object and computes the appropriate reward.
    If place is not good, the object gets removed from the environment, as its state is hard to determine.
    Assumes robot and object are at holding pose.
    - Input descriptor: Location of the hand at place.    
    - Returns reward: 1 if place is on an unoccupied disk and 0 otherwise.
    '''
    
    # parameters
    placeHeightTolerance = self.params["placeHeightTolerance"]
    
    # move object to hand at place
    bTg = self.robot.GetTransform()
    self.MoveHandToPose(descriptor.T)
    self.MoveObjectToHandAtGrasp(bTg, self.holdingObject)  
    if self.params["showSteps"]:
      raw_input("Placed object.")
    self.MoveHandToHoldingPose()
    
    # no longer holding an object
    placedObject = self.holdingObject
    self.holdingObject = None
    self.holdingDescriptor = None
    
    # check if CoM is above a disk
    bTo = placedObject.GetTransform()
    supportObject = None
    for disk in self.supportObjects:
      diskXY = disk.GetTransform()[0:2, 3]
      if norm(diskXY - bTo[0:2, 3]) < disk.radius - placedObject.radius:
        supportObject = disk
        break
    
    # not above any disk
    if supportObject is None:
      self.objects.remove(placedObject)
      self.env.Remove(placedObject)
      return 0.0
    
    # support object is already occupied
    if supportObject in self.placedObjects.values():
      self.objects.remove(placedObject)
      self.env.Remove(placedObject)
      return 0.0
    
    # check if height is good
    supportTopZ = supportObject.GetTransform()[2, 3] + supportObject.height / 2.0
    objectBottomZ = placedObject.GetTransform()[2, 3] - placedObject.height / 2.0    
    if objectBottomZ < supportTopZ - placeHeightTolerance[0] or \
       objectBottomZ > supportTopZ + placeHeightTolerance[1]:
         self.objects.remove(placedObject)
         self.env.Remove(placedObject)
         return 0.0
    
    # place is good
    self.placedObjects[placedObject] = supportObject
    return 1.0

  def PlaceCylinders(self, nObjects, heightMinMax, radiusMinMax, isSupportObjects, \
    maxPlaceAttempts=10, workspace=((-0.18, 0.18), (-0.18, 0.18))):
    '''Places a set of cylinders, with size drawn uniformly at random, on the table, in xy positions
       drawn uniformly at random.
    - Input nObjects: The number of objects to place.
    - Input heightMinMax: Pair of floats specifying range of cylinder height.
    - Input radiusMinMax: Pair of floats specifying range of cylinder radius.
    - Input isSupportObjects: True if objects are ungraspable support objects.
    - Input maxPlaceAttempts: Max reattempts (position samples) if the placement is in collision.
    - Input workspace: xy placement extents: [(xMin, xMax), (yMin, yMax)].
    '''

    objectHandles = []
    for i in xrange(nObjects):

      # create object
      height = uniform(heightMinMax[0], heightMinMax[1])
      radius = uniform(radiusMinMax[0], radiusMinMax[1])
      name = "support-" + str(i+len(self.supportObjects)) if isSupportObjects \
        else "object-" + str(i+len(self.objects))
      geomInfo = openravepy.KinBody.Link.GeometryInfo()
      geomInfo._type = openravepy.KinBody.Link.GeomType.Cylinder
      geomInfo._vGeomData = [radius, height]
      geomInfo._vDiffuseColor = self.colors[randint(len(self.colors))]
      body = openravepy.RaveCreateKinBody(self.env, "")
      body.InitFromGeometries([geomInfo])
      body.SetName(name)
      body.height = height
      body.radius = radius
      self.env.Add(body, True)
      objectHandles.append(body)

      # position object
      for j in xrange(maxPlaceAttempts):

        xy = array([ \
          uniform(workspace[0][0], workspace[0][1]),
          uniform(workspace[1][0], workspace[1][1])])

        T = eye(4)
        T[0:2, 3] = xy
        T[2, 3] = height / 2.0 + self.tableExtents[2] / 2.0
        body.SetTransform(T)

        if not self.env.CheckCollision(body): break

    if isSupportObjects:
      self.supportObjects += objectHandles
    else:
      self.objects += objectHandles

  def ResetEpisode(self):
    '''Resets all internal variables pertaining to a particular episode, including objects placed.'''

    self.RemoveObjectSet(self.objects)
    self.RemoveObjectSet(self.supportObjects)
    self.objects = []
    self.supportObjects = []
    self.holdingObject = None
    self.holdingDescriptor = None
    self.placedObjects = {}

  def Transition(self, descriptor):
    '''The robot takes the provided action and a reward is evaluated.
    The reward is +1 if a mug is placed on an unoccupied coaster, -1 if a mug already placed is
    removed from the coaster, and 0 otherwise.
    - Input descriptor: HandDescriptor object describing the current (overt) action.
    - Returns r: A scalar reward. The state of the objects in the simulator may change.
    '''

    if self.holdingObject is None:
    
      # Test for a grasp.
      
      self.holdingObject = self.IsPegGrasp(descriptor)
      
      if self.holdingObject is None:
        
        # grasp failed
        if self.params["showSteps"]:
          raw_input("Grasp failed.")
        r = 0.0
        
      else:
        
        # grasp succeeded
        if self.params["showSteps"]:
          raw_input("Grasped object.")
        
        r = self.PerformGrasp(descriptor)
    
    else:
      
      # Test for a place.
      
      r = self.PerformPlace(descriptor)

    return self.holdingDescriptor, r