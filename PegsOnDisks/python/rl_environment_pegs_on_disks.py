'''Reinforcement learning (RL) environment for the pegs on disks domain.'''

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

class RlEnvironmentPegsOnDisks(RlEnvironment):

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
    self.placeOrientTolerance = self.params["placeOrientTolerance"]
    self.placeHeightTolerance = self.params["placeHeightTolerance"]
    self.rewardCapGrasps = self.params["rewardCapGrasps"]
    
    self.colors = array([ \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ])
      
    self.pointToRealRadiusError = 0.0001
      
    # initialization
    self.InitializeHandRegions()
    
    self.objectFileNames = os.listdir(self.objectFolder)
    self.objectFileNames = fnmatch.filter(self.objectFileNames, "*.dae")
    self.supportObjectFileNames = os.listdir(self.supportObjectFolder)
    self.supportObjectFileNames = fnmatch.filter(self.supportObjectFileNames, "*.dae")

    # internal state
    self.objects = []
    self.supportObjects = []
    self.ResetEpisode()
    
  def GenerateCylinderMesh(self, heightMinMax, radiusMinMax, name):
    '''Generates a cylinder and saves it into a CAD model file.
    - Input heightMinMax: Tuple specifying range (min, max) from which to select cylinder height.
    - Input radiusMinmax: Tuple specifying range (min, max) from which to select cylinder radius.
    - Input name: String name of object; also determines name of file to save.
    - Returns body: Handle to the openrave object, added to the environment.
    '''

    # create object
    height = uniform(heightMinMax[0], heightMinMax[1])
    radius = uniform(radiusMinMax[0], radiusMinMax[1])
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
    
    # save mesh file
    self.env.Save(name + ".dae", openravepy.Environment.SelectionOptions.Body, name)
    print("Saved " + name + ".")
    
    return body
    
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

  def IsPegGrasp(self, descriptor):
    '''Checks if, when the hand is placed at the descriptor's pose and closed, a grasp takes place.
       A grasp must be (1) collision-free (2) contain exactly 1 peg's geometry, (3) contain the
       cylinder's axis, and (4) not contact the side and cap of the cylinder.
    - Input descriptor: HandDescriptor object of the target hand pose.
    - Returns graspedObject: The handle of the grasped object if a cylinder can be grasped from the
      target hand pose; otherwise None.
    - Returns isCapGrasp: True if this is a good grasp and each finger contacts the bottom/top of
      the peg.
    '''

    # check collision
    collision, objCloudsInHandFrame = self.IsRobotInCollision(descriptor)
    if collision: return None, False
    
    # check intersection of exactly 1 object
    graspedObject = None; pointsInHand = None
    for i, obj in enumerate(self.objects):
      X = point_cloud.FilterWorkspace(self.handClosingRegion, objCloudsInHandFrame[i])
      intersect = X.size > 0
      if intersect:
        if graspedObject is None:
          graspedObject = obj
          pointsInHand = X
        else:
          # intersection of multiple objects
          return None, False
          
    if graspedObject is None:
      # intersection of no objects
      return None, False
    
    # A cylinder can only be upright or on the side. We handle these two cases separately.
    bTo = graspedObject.GetTransform()
    
    if self.IsPegUpright(graspedObject):
      # Top-center of cylinder in the hand is necessary and sufficient.
      bp = copy(bTo[0:3, 3])
      bp[2] += graspedObject.height / 2.0
      hp = point_cloud.Transform(inv(descriptor.T), array([bp]))
      hP = point_cloud.FilterWorkspace(self.handClosingRegion, hp)
      if hP.size == 0:
        return None, False
      return graspedObject, False

    # Cylinder is on its side.

    # check if finger tips are below cylinder axis
    cylinderZ = bTo[2, 3]
    fingerZ = descriptor.center[2] - descriptor.depth / 2.0
    if fingerZ > cylinderZ:
      return None, False
    
    # make sure cylinder caps are not in hand
    contactIdxs = array([argmax(pointsInHand[:, 1]), argmin(pointsInHand[:, 1])])
    contacts = pointsInHand[contactIdxs, :]
    oX = point_cloud.Transform(dot(inv(bTo), descriptor.T), pointsInHand)
    capIdxs = sum(power(oX[:, 0:2], 2), 1) < (graspedObject.radius - self.pointToRealRadiusError)**2
    capIdxs = capIdxs.flatten()   
    nContactsOnCap = sum(capIdxs[contactIdxs])
    if nContactsOnCap == 1 or sum(power(contacts[0, 0:2] - contacts[1, 0:2], 2)) < \
      (min(2 * graspedObject.radius, graspedObject.height) - 2 * self.pointToRealRadiusError)**2:
      # 1 finger contacts cap, other finger contacts side
      return None, False
    
    # side grasp is good
    return graspedObject, nContactsOnCap == 2

  def IsRobotInCollision(self, descriptor):
    '''Checks collision between the robot and the world.
    - Input descriptor: HandDescriptor object for the current hand pose.
    - Returns: True if in collision and False otherwise.
    - Returns objCloudsInHandFrame: List of point clouds, one for each object, in the descriptor
      reference frame. Or, None if a collision is detected. (This is to avoid performing transforms
      of all object clouds twice.)
    '''

    # ODE misses several box-cylinder collisions. So we have to implement this ourselves.
    
    # check collision with table
    bX = point_cloud.Transform(descriptor.T, self.externalHandPoints)
    if (bX[:, 2] < self.GetTableHeight()).any():
      return True, None
    
    # some preparation
    hTb = inv(descriptor.T)
    self.robot.SetTransform(descriptor.T) # for visualization only
    objects = self.objects + self.supportObjects
    objCloudsInHandFrame = []
    
    # check if any object points intersect hand collision geometry
    for i, obj in enumerate(objects):
      bTo = obj.GetTransform()
      hX = point_cloud.Transform(dot(hTb, bTo), obj.cloud)
      X = point_cloud.FilterWorkspace(self.handFingerRegionL, hX)
      if X.size > 0: return True, None
      X = point_cloud.FilterWorkspace(self.handFingerRegionR, hX)
      if X.size > 0: return True, None
      X = point_cloud.FilterWorkspace(self.handTopRegion, hX)
      if X.size > 0: return True, None
      if i < len(self.objects): objCloudsInHandFrame.append(hX)

    return False, objCloudsInHandFrame
  
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
      
  def IsPegUpright(self, obj):
    '''Returns True iff the peg's axis is (nearly) normal to the table plane. In this environment it
      can be only be normal or orthogonal.'''
    
    return abs(obj.GetTransform()[2, 2]) > 0.9
    
  def PerformGrasp(self, descriptor, cloud):
    '''Tests for and simulates a grasp. If an object is grasped, self.holdingObject is set.
    - Input descriptor: Pose of the grasp.
    - Input cloud: Point cloud of the current scene, in the base/world frame (excluding table).
    - Returns reward: -1 if grasping a placed object, 1 if grasping an unplaced object, and 0 otherwise.
    '''
    
    self.holdingObject, isCapGrasp = self.IsPegGrasp(descriptor)
    
    if not self.holdingObject:
      if self.params["showSteps"]:
        raw_input("Grasp failed.")
      return 0.0
    
    if self.params["showSteps"]:
      raw_input("Grasp succeeded.")
    
    # generate grasp image
    descriptor.GenerateHeightmap(cloud, self.GetTableHeight())
    self.holdingDescriptor = descriptor
    
    # simulate object movement when hand closes
    self.SimulateObjectMovementOnClose(descriptor, self.holdingObject, isCapGrasp)
    
    # move to holding pose
    self.MoveHandToHoldingPose()
    self.MoveObjectToHandAtGrasp(descriptor.T, self.holdingObject)
    
    # compute reward
    if self.holdingObject in self.placedObjects:
      del self.placedObjects[self.holdingObject]
      return -1.0
    if not self.rewardCapGrasps and isCapGrasp:
      return 0.0
    return 1.0
    
  def PerformPlace(self, descriptor):
    '''Places the object and computes the appropriate reward. If place is not good, the object gets
      removed from the environment, as its resulting state is hard to determine. Assumes robot and
      object are at the holding pose.
    - Input descriptor: Location of the hand at place.    
    - Returns reward: 1 if place is on an unoccupied disk and 0 otherwise.
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
    
    # check if peg is vertical
    bTo = placedObject.GetTransform()
    if abs(dot(bTo[0:3, 2], array([0.0, 0.0, 1.0]))) < 1.0 - self.placeOrientTolerance:
      self.PlaceFailed(placedObject)
      return 0.0
    
    # check if peg is entirely over a disk
    supportObject = None
    for disk in self.supportObjects:
      diskXY = disk.GetTransform()[0:2, 3]
      if sum(power(diskXY - bTo[0:2, 3], 2)) < (disk.radius - placedObject.radius)**2:
        supportObject = disk
        break
    
    # not above any disk
    if supportObject is None:
      self.PlaceFailed(placedObject)
      return 0.0
    
    # support object is already occupied
    if supportObject in self.placedObjects.values():
      self.PlaceFailed(placedObject)
      return 0.0
    
    # check if height is good
    supportTopZ = supportObject.GetTransform()[2, 3] + supportObject.height / 2.0
    objectBottomZ = placedObject.GetTransform()[2, 3] - placedObject.height / 2.0    
    if objectBottomZ < supportTopZ - self.placeHeightTolerance[0] or \
       objectBottomZ > supportTopZ + self.placeHeightTolerance[1]:
         self.PlaceFailed(placedObject)
         return 0.0
         
    # check if hand is in collision
    collision, cloudsInHandFrame = self.IsRobotInCollision(descriptor)
    if collision:
      self.PlaceFailed(placedObject)
      return 0.0
    
    # place is good
    if self.params["showSteps"]:
      raw_input("Placed object successfully.")
    self.placedObjects[placedObject] = supportObject
    return 1.0
    
  def PlaceObjects(self, isSupportObjects, maxPlaceAttempts=10,
    workspace=((-0.18, 0.18), (-0.18, 0.18))):
    '''Chooses and places objects randomly on the table.
    - Input isSupportObjects: Are the objects support objects (i.e. disks)?
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
      self.env.Load(folderName + "/" + objectName)
      shortObjectName = objectName[:-4]
      body = self.env.GetKinBody(shortObjectName)

      # load points, height, and radius
      data = loadmat(folderName + "/" + shortObjectName + ".mat")
      body.cloud = data["cloud"]
      body.height = data["height"]
      body.radius = data["radius"]
      
      # select pose for object
      for j in xrange(maxPlaceAttempts):

        # choose orientation        
        r1 = 0.0 if isSupportObjects else choice([pi / 2.0, 0.0], p=[2.0 / 3.0, 1.0 / 3.0])
        r2 = uniform(0, 2.0 * pi)
        R1 = openravepy.matrixFromAxisAngle([1.0, 0.0, 0.0], r1)
        R2 = openravepy.matrixFromAxisAngle([0.0, 1.0, 0.0], r2) if r1 > 0 else eye(4)
        
        # choose xy position
        xy = array([ \
          uniform(workspace[0][0], workspace[0][1]),
          uniform(workspace[1][0], workspace[1][1])])
        
        # set height
        z = body.height / 2.0 if r1 == 0 else copy(body.radius)
        z += self.GetTableHeight()
        
        # set transform
        T = eye(4)
        T[0:2, 3] = xy
        T[2, 3] = z
        T = dot(T, dot(R1, R2))
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
    
  def SimulateObjectMovementOnClose(self, descriptor, obj, isCapGrasp):
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
    
    if self.IsPegUpright(obj):
      # Top grasp. Simply set the y-position to 0.
      hTo[1, 3] = 0
    elif isCapGrasp:
      # Side grasp where fingers contact peg caps.
      # Set y = 0 at center point.
      hTo[1, 3] = 0
      # Set the orientation to be horizontal in hand.
      zAxis = hTo[0:2, 2] if hTo[1, 2] >= 0 else -hTo[0:2, 2]
      angle = arccos(dot(zAxis, array([0.0, 1.0])) / norm(zAxis))
      angle = angle if zAxis[0] >= 0 else 2 * pi - angle
      handDepthAxis = array([0.0, 0.0, 1.0])
      T = openravepy.matrixFromAxisAngle(handDepthAxis, angle)
      hTo = dot(T, hTo)
    else:
      # Side grasp.
      # Set y = 0 at the point of contact along the cylinder axis.
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
    The reward is +1 if a peg is placed on an unoccupied disk, -1 if a peg already placed is
    removed from a disk, and 0 otherwise.
    - Input descriptor: HandDescriptor object describing the current (overt) action.
    - Input cloud: Point cloud of the scene, excluding table.
    - Returns r: A scalar reward. The state of the objects in the simulator may change.
    '''

    if self.holdingObject is None:
      r = self.PerformGrasp(descriptor, cloud)
    else:
      r = self.PerformPlace(descriptor)

    return self.holdingDescriptor, r