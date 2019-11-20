'''Reinforcement learning (RL) environment for the bottles on coasters domain.'''

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
from numpy import arccos, argmax, argmin, array, arange, cos, dot, eye, hstack, logical_or, mean, \
  pi, power, repeat, reshape, sin, sqrt, sum, vstack, zeros
# openrave
import openravepy
# self
import point_cloud
from rl_environment import RlEnvironment
from hand_descriptor import HandDescriptor

class RlEnvironmentBottlesOnCoasters(RlEnvironment):

  def __init__(self, params):
    '''Initializes openrave environment, parameters, and first episode.
    - Input params: System parameters data structure.
    '''

    RlEnvironment.__init__(self, params)

    # parameters
    self.nObjects = params["nObjects"]
    self.nSupportObjects = params["nSupportObjects"]
    self.objectFolder = params["objectFolder"]
    self.supportObjectFolder = params["supportObjectFolder"]
    self.graspFrictionCone = params["graspFrictionCone"]
    self.graspMinDistFromBottom = params["graspMinDistFromBottom"]
    self.placeOrientTolerance = self.params["placeOrientTolerance"]
    self.placePosTolerance = self.params["placePosTolerance"]
    self.placeHeightTolerance = self.params["placeHeightTolerance"]
      
    # initialization
    self.InitializeHandRegions()
    self.objectFileNames = os.listdir(self.objectFolder)
    self.objectFileNames = fnmatch.filter(self.objectFileNames, "*.dae")
    self.supportObjectFileNames = os.listdir(self.supportObjectFolder)
    self.supportObjectFileNames = fnmatch.filter(self.supportObjectFileNames, "*.dae")

    # internal state
    self.objects = []
    self.supportObjects = []
    self.kinBodyCache = {}
    self.ResetEpisode()
    
  def GetArtificialCloud(self):
    '''Concatenates point cloud data from all objects and support objects.
    - Returns cloud: Point cloud in the base/world reference frame.
    '''
    
    clouds = []
    objects = self.supportObjects + self.objects
    
    for obj in objects:
      cloud = point_cloud.Transform(obj.GetTransform(), obj.cloud)
      clouds.append(cloud)

    return vstack(clouds)
    
  def InitializeHandRegions(self):
    '''Determines hand geometry, in the descriptor reference frame, for collision checking. Should
    be called once at initialization.
  '''
    
    # find default descriptor geometry
    desc = HandDescriptor(eye(4), self.params["imP"], self.params["imD"], self.params["imW"])
    
    # important reference points
    topUp = desc.top + (desc.height / 2) * desc.axis
    topDn = desc.top - (desc.height / 2) * desc.axis
    BtmUp = desc.top + (desc.height / 2) * desc.axis
    BtmDn = desc.top - (desc.height / 2) * desc.axis
    
    # cuboids representing hand regions, in workspace format
    self.handClosingRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width  / 2, desc.width  / 2),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handFingerRegionL = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, -desc.width  / 2),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handFingerRegionR = [
      (-desc.height / 2, desc.height / 2),
      (desc.width / 2, desc.width / 2 + 0.01),
      (-desc.depth  / 2, desc.depth  / 2)]
      
    self.handTopRegion = [
      (-desc.height / 2, desc.height / 2),
      (-desc.width / 2 - 0.01, desc.width / 2 + 0.01),
      (desc.depth  / 2, desc.depth  / 2 + 0.01)]
    
    # find corners of hand collision geometry
    self.externalHandPoints = array([ \
      topUp + ((desc.width / 2) + 0.01) * desc.binormal,
      topUp - ((desc.width / 2) + 0.01) * desc.binormal,
      topDn + ((desc.width / 2) + 0.01) * desc.binormal,
      topDn - ((desc.width / 2) + 0.01) * desc.binormal,
      BtmUp + ((desc.width / 2) + 0.01) * desc.binormal,
      BtmUp - ((desc.width / 2) + 0.01) * desc.binormal,
      BtmDn + ((desc.width / 2) + 0.01) * desc.binormal,
      BtmDn - ((desc.width / 2) + 0.01) * desc.binormal, ])
    
  def IsAntipodalGrasp(self, descriptor, targetObject, maxAngleToFinger):
    '''Returns True if a grasp is near antipodal, based on the parameters.
    - Input descriptor: HandDescriptor object with pose of grasp.
    - Input targetObject: OpenRAVE object handle with cloud and normals entries.
    - Input maxAngleToFinger: Maximum angle between surfance normal and finger in degrees. Used
      10 degrees for blocks, 15 degrees for mugs, and 15 degrees for bottles.
    - Returns: True if antipodal grasp, False otherwise.
    '''

    # parameters
    contactWidth = 0.01
    maxAntipodalDist = 0.01
    maxAngleToFinger = cos(maxAngleToFinger*(pi/180))

    # put cloud into hand reference frame
    bTo = targetObject.GetTransform()
    bTh = descriptor.T
    hTo = dot(inv(bTh), bTo)
    X, N = point_cloud.Transform(hTo, targetObject.cloud, targetObject.normals)
    X, N = point_cloud.FilterWorkspace(self.handClosingRegion, X, N)
    if X.shape[0] == 0:
      #print("No points in hand.")
      return False

    # find contact points
    leftPoint = min(X[:, 1]); rightPoint = max(X[:, 1])
    lX, lN = point_cloud.FilterWorkspace([(-1,1),(leftPoint,leftPoint+contactWidth),(-1,1)], X, N)
    rX, rN = point_cloud.FilterWorkspace([(-1,1),(rightPoint-contactWidth,rightPoint),(-1,1)], X, N)

    # find contact points normal to finger
    lX = lX[-lN[:, 1] >= maxAngleToFinger, :]
    rX = rX[ rN[:, 1] >= maxAngleToFinger, :]
    if lX.shape[0] == 0 or rX.shape[0] == 0:
      #print("No contact points normal to finger.")
      return False

    # are the closest two contact points nearly antipodal?
    leftTree = cKDTree(lX[:,(0, 2)])
    d, idxs = leftTree.query(rX[:, (0,2)])
    #if min(d) >= maxAntipodalDist:
    #  print("Contacts not antipodal.")
    return min(d) < maxAntipodalDist

  def IsGrasp(self, descriptor):
    '''Checks if, when the hand is placed at the descriptor's pose and closed, a grasp takes place.
    - Input descriptor: HandDescriptor object of the target hand pose.
    - Returns graspedObject: The handle of the grasped object if a cylinder can be grasped from the
      target hand pose; otherwise None.
    '''

    # check collision
    if self.IsRobotInCollision(descriptor):
      return None
      
    # check intersection of exactly 1 object
    graspedObject = None
    for i, obj in enumerate(self.objects):
      bTo = obj.GetTransform()
      hTo = dot(inv(descriptor.T), bTo)
      X = point_cloud.Transform(hTo, obj.cloud)
      X = point_cloud.FilterWorkspace(self.handClosingRegion, X)
      intersect = X.size > 0
      if intersect:
        if graspedObject is None:
          graspedObject = obj
        else:
          # intersection of multiple objects
          return None
          
    if graspedObject is None:
      # intersection of no objects
      return None
      
    # check antipodal condition
    if self.IsAntipodalGrasp(descriptor, graspedObject, self.graspFrictionCone):
      return graspedObject
    return None

  def IsRobotInCollision(self, descriptor):
    '''Checks collision between the robot and the world.
    - Input descriptor: HandDescriptor object for the current hand pose.
    - Returns: True if in collision and False otherwise.
    '''
    
    self.robot.SetTransform(descriptor.T)
    return self.env.CheckCollision(self.robot)
      
  def IsBottleUpright(self, obj):
    '''Returns True iff the bottle's axis is (nearly) normal to the table plane. In this environment
      it can be only be normal or orthogonal.'''
    
    return dot(obj.GetTransform()[0:3, 2], array([0.0, 0.0, 1.0])) > \
      cos(self.placeOrientTolerance * pi / 180)
    
  def PerformGrasp(self, descriptor, cloud):
    '''Tests for and simulates a grasp. If an object is grasped, self.holdingObject is set.
    - Input descriptor: Pose of the grasp.
    - Input cloud: Point cloud of the current scene, in the base/world frame (excluding table).
    - Returns reward: -1 if grasping a placed object, 1 if grasping an unplaced object, and 0 otherwise.
    '''
    
    self.holdingObject = self.IsGrasp(descriptor)
    
    if not self.holdingObject:
      if self.params["showSteps"]:
        raw_input("Grasp failed.")
      return 0.0
    
    if self.params["showSteps"]:
      raw_input("Grasp succeeded.")
    
    # penalize grasps near bottom of bottle (kinematics consideration)
    oTd = dot(inv(self.holdingObject.GetTransform()), descriptor.T)
    graspTooLow = oTd[2, 3] - self.holdingObject.workspace[2][0] < self.graspMinDistFromBottom
    
    # generate grasp image
    descriptor.GenerateHeightmap(cloud, self.GetTableHeight())
    self.holdingDescriptor = descriptor
    
    # simulate object movement when hand closes
    self.SimulateObjectMovementOnClose(descriptor, self.holdingObject)
    
    # move to holding pose
    self.MoveHandToHoldingPose()
    self.MoveObjectToHandAtGrasp(descriptor.T, self.holdingObject)
    
    # compute reward
    if self.holdingObject in self.placedObjects:
      del self.placedObjects[self.holdingObject]
      return -1.0
    
    if graspTooLow:
      return 0.0
    
    return 1.0
    
  def PerformPlace(self, descriptor):
    '''Places the object and computes the appropriate reward. If place is not good, the object gets
      removed from the environment, as its resulting state is hard to determine. Assumes robot and
      object are at the holding pose.
    - Input descriptor: Location of the hand at place.    
    - Returns reward: 1 if place is on an unoccupied coaster and 0 otherwise.
    '''
    
    # move object to hand at place
    bTg = self.robot.GetTransform()
    self.MoveHandToPose(descriptor.T)
    self.MoveObjectToHandAtGrasp(bTg, self.holdingObject)  
    self.MoveHandToHoldingPose()
    
    # no longer holding an object
    placedObject = self.holdingObject
    self.holdingObject = None
    self.holdingDescriptor = None
    
    # check if bottle is vertical
    if not self.IsBottleUpright(placedObject):
      self.PlaceFailed(placedObject)
      return 0.0
    
    # check if bottle is approximately completely over the coaster
    bTo = placedObject.GetTransform()
    supportObject = None
    for coaster in self.supportObjects:
      coasterXY = coaster.GetTransform()[0:2, 3]
      if sum(power(coasterXY - bTo[0:2, 3], 2)) < (coaster.radius - self.placePosTolerance)**2:
        supportObject = coaster
        break
    
    # not above any coaster
    if supportObject is None:
      self.PlaceFailed(placedObject)
      return 0.0  
    
    # support object is already occupied
    if supportObject in self.placedObjects.values():
      self.PlaceFailed(placedObject)
      return 0.0
    
    # check if bottle bottom is within given height tolerance
    supportTopZ = supportObject.GetTransform()[2, 3] + supportObject.workspace[2, 1]
    objectBottomZ = placedObject.GetTransform()[2, 3] + placedObject.workspace[2, 0] 
    if objectBottomZ < supportTopZ - self.placeHeightTolerance[0] or \
       objectBottomZ > supportTopZ + self.placeHeightTolerance[1]:
         self.PlaceFailed(placedObject)
         return 0.0
    
    '''# check if hand is in collision
    collision, cloudsInHandFrame = self.IsRobotInCollision(descriptor)
    if collision:
      self.PlaceFailed(placedObject)
      return 0.0'''
    
    # place is good
    if self.params["showSteps"]:
      raw_input("Placed object successfully.")
    self.placedObjects[placedObject] = supportObject
    return 1.0
    
  def PlaceObjects(self, isSupportObjects, maxPlaceAttempts=10,
    workspace=((-0.18, 0.18), (-0.18, 0.18))):
    '''Chooses and places objects randomly on the table.
    - Input isSupportObjects: Are the objects support objects (i.e. coasters)?
    - Input maxPlaceAttempts: Maximum number of times to place an object collision-free. If exceeded,
      the object will be placed in collision with some already placed object(s).
    - Input workspace: Area to place objects in, [(minX, maxX), (minY, maxY)]. Center of objects will
      not be outside of these bounds.
    - Returns None.
    '''
    
    # support object / graspable object
    if isSupportObjects:
      nObjects = self.nSupportObjects
      folderName = self.supportObjectFolder
      fileNames = self.supportObjectFileNames
    else:
      nObjects = self.nObjects
      folderName = self.objectFolder
      fileNames = self.objectFileNames

    # select file(s)
    fileIdxs = choice(len(fileNames), size=nObjects, replace=False)
    objectHandles = []

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
        if "normals" in data: body.normals = data["normals"]
        if "radius" in data: body.radius = data["radius"]
        # add to cache
        self.kinBodyCache[objectName] = body
      
      # select pose for object
      for j in xrange(maxPlaceAttempts):
        
        # choose orientation
        if isSupportObjects:
          R = eye(4)
          downAxis = (2, 0)
        else:
          r1 = choice(array([0, 1, 2, 3]) * (pi / 2))
          r2 = choice([pi / 2.0, 0.0], p=[2.0 / 3.0, 1.0 / 3.0])
          r3 = uniform(0, 2 * pi)
          R1 = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], r1)
          R2 = openravepy.matrixFromAxisAngle([0.0, 1.0, 0.0], r2)
          R3 = openravepy.matrixFromAxisAngle([0.0, 0.0, 1.0], r3)
          R = dot(R3, dot(R2, R1)) # about fixed frame in order 1, 2, 3
          
          if r2 == 0:
            downAxis = (2, 0)
          elif r1 == 0:
            downAxis = (0, 1)
          elif r1 == pi/2:
            downAxis = (1, 0)
          elif r1 == pi:
            downAxis = (0, 0)
          else:
            downAxis = (1, 1)
        
        # choose xy position
        xy = array([ \
          uniform(workspace[0][0], workspace[0][1]),
          uniform(workspace[1][0], workspace[1][1])])
        
        # set height
        z = abs(body.workspace[downAxis]) + self.GetTableHeight() + 0.001
        
        # set transform
        T = eye(4)
        T[0:2, 3] = xy
        T[2, 3] = z
        T = dot(T, R)
        body.SetTransform(T)
        
        if not self.env.CheckCollision(body): break

      # add to environment
      objectHandles.append(body)

    if isSupportObjects:
      self.supportObjects += objectHandles
    else:
      self.objects += objectHandles
      
  def PlaceFailed(self, placedObject):
    '''Helper function to be called if a successful place condition is not met.'''
    
    if self.params["showSteps"]:
      raw_input("Place failed.")
    self.objects.remove(placedObject)
    self.env.Remove(placedObject)

  def ResetEpisode(self):
    '''Resets all internal variables pertaining to a particular episode, including objects placed.'''

    self.RemoveObjectSet(self.objects)
    self.RemoveObjectSet(self.supportObjects)
    self.objects = []
    self.supportObjects = []
    self.holdingObject = None
    self.holdingDescriptor = None
    self.placedObjects = {}
    
  def SimulateObjectMovementOnClose(self, descriptor, obj):
    '''The object can move when the fingers close during a grasp.
      This sets the object to an approximation to the correct resultant pose.
    - Input descriptor: Grasp pose. Assumes this is a valid grasp.
    - Input obj: The object being grasped.
    - Returns None.
    '''
    
    # get object pose in hand frame
    bTo = obj.GetTransform()
    hTb = inv(descriptor.T)
    hTo = dot(hTb, bTo)
    
    if self.IsBottleUpright(obj):
      # Top grasp. Simply set the y-position to 0.
      hTo[1, 3] = 0
    else:
      # Side grasp.
      # Set y = 0 at the point of contact along the bottle axis.
      alpha = -hTo[0, 3] / hTo[0, 2]
      deltaY = hTo[1, 3] + alpha * hTo[1, 2]
      hTo[1, 3] -= deltaY
      # Set the orientation to be vertical in the hand.
      zAxis = hTo[0:2, 2] if hTo[0, 2] >= 0 else -hTo[0:2, 2]
      angle = arccos(dot(zAxis, array([1.0, 0.0])) / norm(zAxis))
      angle = angle if zAxis[1] <= 0 else 2 * pi - angle
      handDepthAxis = array([0.0, 0.0, 1.0])
      T = openravepy.matrixFromAxisAngle(handDepthAxis, angle)
      hTo = dot(T, hTo)
    
    # set the object's new pose in the base frame
    bToNew = dot(descriptor.T, hTo)
    obj.SetTransform(bToNew)

  def Transition(self, descriptor, cloud):
    '''The robot takes the provided action and a reward is evaluated.
    The reward is +1 if a bottle is placed on an unoccupied coaster, -1 if a bottle already placed
    is removed from a coaster, and 0 otherwise.
    - Input descriptor: HandDescriptor object describing the current (overt) action.
    - Input cloud: Point cloud of the scene, excluding table.
    - Returns r: A scalar reward. The state of the objects in the simulator may change.
    '''

    if self.holdingObject is None:
      r = self.PerformGrasp(descriptor, cloud)
    else:
      r = self.PerformPlace(descriptor)

    return self.holdingDescriptor, r