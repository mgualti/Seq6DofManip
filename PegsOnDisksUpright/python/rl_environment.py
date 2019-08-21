'''Reinforcement learning (RL) environment.'''

# python
import os
import fnmatch
from time import sleep, time
# scipy
from scipy.io import loadmat
from scipy.spatial import cKDTree
from numpy.linalg import inv, norm
from numpy.random import choice, rand, randint, randn, uniform
from numpy import array, cos, cross, dot, eye, hstack, ones, pi, sin, tile, vstack, zeros
# openrave
import openravepy
# self
import point_cloud

class RlEnvironment:

  def __init__(self, params):
    '''Initializes openrave environment, etc.'''

    # Parameters
    self.params = params
    self.showViewer = params["showViewer"]
    self.projectDir = os.getcwd() + "/"

    # Create openrave environment
    self.env = openravepy.Environment()
    if params["showViewer"]: self.env.SetViewer('qtcoin')
    self.env.Load(self.projectDir + "openrave/environment.xml")
    self.robot = self.env.GetRobots()[0]
    self.robot.SetDOFValues(array([0.0475]))

    # set collision checker options
    collisionChecker = openravepy.RaveCreateCollisionChecker(self.env, 'ode')
    self.env.SetCollisionChecker(collisionChecker)

    # don't want to be affected by gravity, since it is floating
    for link in self.robot.GetLinks():
      link.SetStatic(True)

    # set physics options
    #self.physicsEngine = openravepy.RaveCreatePhysicsEngine(self.env, "ode")
    #self.env.SetPhysicsEngine(self.physicsEngine)
    #self.env.GetPhysicsEngine().SetGravity([0,0,-9.8])
    self.env.StopSimulation()

    # table(s)
    self.tableObj = self.env.GetKinBody("table")
    self.tablePosition = self.tableObj.GetTransform()[0:3, 3]
    self.tableExtents = self.tableObj.ComputeAABB().extents()
    if params["removeTable"]:
      self.env.Remove(self.tableObj)

    # Internal state
    self.objectPoses = []
    self.plotCloudHandle = None
    self.plotDescriptorsHandle = None

  def MoveHandToHoldingPose(self):
    '''Moves the hand to a special, pre-designated holding area.'''

    T = eye(4)
    T[0:3, 3] = array([-1.00, 0,  0.30])

    self.MoveHandToPose(T)

  def MoveHandToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.robot.SetTransform(T)
    self.env.UpdatePublishedBodies()

  def MoveObjectToHandAtGrasp(self, bTg, objectHandle):
    '''Aligns the grasp on the object to the current hand position and moves the object there.
      - Input: The grasp in the base frame (4x4 homogeneous transform).
      - Input objectHandle: Handle to the object to move.
      - Retruns X: The transform applied to the object.
    '''

    bTo = objectHandle.GetTransform()
    bTr = self.robot.GetTransform()

    X = dot(bTr, inv(bTg))
    objectHandle.SetTransform(dot(X, bTo))

    return X

  def PlotCloud(self, cloud):
    '''Plots a cloud in the environment.'''

    if not self.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.UnplotCloud()

    self.plotCloudHandle = self.env.plot3(\
      points=cloud, pointsize=0.001, colors=zeros(cloud.shape), drawstyle=1)

  def PlotDescriptors(self, descriptors, graspColorRgb=[1,0,0]):
    '''Visualizes grasps in openrave viewer.'''

    if not self.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.UnplotDescriptors()

    if len(descriptors) == 0:
      return

    lineList = []; colorList = []
    for desc in descriptors:

      c = desc.bottom
      a = c - desc.depth*desc.approach
      l = c - 0.5*desc.width*desc.binormal
      r = c + 0.5*desc.width*desc.binormal
      lEnd = l + desc.depth*desc.approach
      rEnd = r + desc.depth*desc.approach

      lineList.append(c); lineList.append(a)
      lineList.append(l); lineList.append(r)
      lineList.append(l); lineList.append(lEnd)
      lineList.append(r); lineList.append(rEnd)

      for i in xrange(8): colorList.append(graspColorRgb)

    self.plotDescriptorsHandle = self.env.drawlinelist(\
      points=array(lineList), linewidth=3.0, colors=array(colorList))

  def RemoveObjectSet(self, objectHandles):
    '''Removes all of the objects in the list objectHandles.'''

    with self.env:
      for objectHandle in objectHandles:
        self.env.Remove(objectHandle)

  def ResetObjectPoses(self, objectHandles):
    '''Replaces all objects to their remembered poses.'''

    with self.env:
      for i, obj in enumerate(objectHandles):
        self.objectPoses.append(obj.SetTransform(self.objectPoses[i]))

  def SetObjectPoses(self, objectHandles):
    '''Saves the pose of all objects in the scene to memory.'''

    with self.env:
      self.objectPoses = []
      for obj in objectHandles:
        self.objectPoses.append(obj.GetTransform())

  def UnplotCloud(self):
    '''Removes a cloud from the environment.'''

    if not self.showViewer:
      return

    if self.plotCloudHandle is not None:
      self.plotCloudHandle.Close()
      self.plotCloudHandle = None

  def UnplotDescriptors(self):
    '''Removes any descriptors drawn in the environment.'''

    if not self.showViewer:
      return

    if self.plotDescriptorsHandle is not None:
      self.plotDescriptorsHandle.Close()
      self.plotDescriptorsHandle = None