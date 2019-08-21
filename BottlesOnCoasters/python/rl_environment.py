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
    
    self.colors = array([ \
      (1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5), (0.0, 1.0, 1.0 ,0.5),
      (1.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5), (0.5, 0.0, 1.0, 0.5),
      (0.0, 0.5, 1.0, 0.5), (1.0, 0.5, 0.0, 0.5), (1.0, 0.0, 0.5, 0.5), (0.0, 1.0, 0.5, 0.5)  ])

    # Create openrave environment
    self.env = openravepy.Environment()
    if params["showViewer"]: self.env.SetViewer('qtcoin')
    self.env.Load(self.projectDir + "openrave/environment.xml")
    self.robot = self.env.GetRobots()[0]
    config = self.robot.GetDOFValues()
    config[-1] = 0.0475  # open gripper
    self.robot.SetDOFValues(config)
    self.manip = self.robot.GetActiveManipulator()

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
    
    # get sensor from openrave
    self.sensor = self.env.GetSensors()[0]
    self.sTh = dot(inv(self.sensor.GetTransform()), self.robot.GetTransform())

    # table(s)
    self.tableObj = self.env.GetKinBody("table")
    self.tablePosition = self.tableObj.GetTransform()[0:3, 3]
    self.tableExtents = self.tableObj.ComputeAABB().extents()
    self.tableHeight = self.tablePosition[2] + self.tableExtents[2] / 2.0
    if params["removeTable"]:
      self.env.Remove(self.tableObj)

    # Internal state
    self.objectPoses = []
    self.plotCloudHandle = None
    self.plotDescriptorsHandle = None
    
  def GenerateMeshFromMesh(self, originalMeshDirectory, scaleMinMax, fileNamesToSkip, name):
    '''Loads a random mesh from a ply file, applies a random scaling and color, and saves as dae.
    - Input originalMeshDirectory: The directory from which to select meshes.
    - Input scaleMinMax: Range [min, max] from which to sample scales uniformly at random.
    - Input fileNamesToSkip: List of file names in originalMeshDirectory to avoid selecting.
    - Input name: Name of the new object (determines save file name).
    - Returns body: OpenRAVE handle to object. Object will be in the environment.
    '''
    
    # load objet
    fileNames = os.listdir(originalMeshDirectory)
    fileNames = fnmatch.filter(fileNames, "*.ply")
    fileName = None
    while fileName is None or fileName in fileNamesToSkip:
      fileName = fileNames[randint(len(fileNames))]
    scale = uniform(scaleMinMax[0], scaleMinMax[1])
    self.env.Load(originalMeshDirectory + "/" + fileName, {'scalegeometry':str(scale)})
    body = self.env.GetKinBody(fileName[:-4])

    # set object properties
    body.SetName(name)
    body.scale = scale
    color = self.colors[randint(len(self.colors))]
    for geom in body.GetLinks()[0].GetGeometries():
      geom.SetDiffuseColor(color)

    # it's also possible to save as ply if ctmconv is installed
    self.env.Save(name + ".dae", openravepy.Environment.SelectionOptions.Body, name)
    print("Saved " + name + ".dae.")

    return body
    
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
    print("Saved " + name + ".dae.")
    
    return body

  def GetCloud(self, workspace=None):
    '''Renders point cloud from the current sensor position.'''

    self.StartSensor()
    self.env.StepSimulation(0.001)

    data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
    cloud = data.ranges + data.positions[0]

    self.StopSensor()

    if workspace is not None:
      cloud = point_cloud.FilterWorkspace(workspace, cloud)

    return cloud

  def GetFullCloud(self, viewCenter, viewKeepout, workspace, add45DegViews=False,
    computeNormals=False, voxelSize=None):
    '''Gets a full point cloud of the scene (6 views) and also computes normals.'''

    poses = self.GetFullViewPoses(viewCenter, viewKeepout, add45DegViews)

    cloud = []; viewPoints = []
    for pose in poses:
      self.MoveSensorToPose(pose)
      X = self.GetCloud(workspace)
      cloud.append(X)
      V = tile(pose[0:3, 3], (X.shape[0], 1))
      viewPoints.append(V)

    cloud = vstack(cloud)
    viewPoints = vstack(viewPoints)
    
    if computeNormals:      
      normals = point_cloud.ComputeNormals(cloud, viewPoints, kNeighbors=30, rNeighbors=-1)
      if voxelSize: cloud, normals = point_cloud.Voxelize(voxelSize, cloud, normals)
      return cloud, normals
    
    if voxelSize: cloud = point_cloud.Voxelize(voxelSize, cloud)
    return cloud
    
  def GetFullViewPoses(self, viewCenter, viewKeepout, add45DegViews):
    '''Returns 6 poses, covering the full object. (No face has incidence more than 90 degrees.)'''

    viewPoints = []
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0,  1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  0, -1]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0,  1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 0, -1,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([ 1,  0,  0]))
    viewPoints.append(viewCenter + viewKeepout*array([-1,  0,  0]))

    if add45DegViews:
      viewPoints.append(viewCenter + viewKeepout*array([0, -cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([0,  cos(45*(pi/180)), sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([-cos(45*(pi/180)), 0, sin(45*(pi/180))]))
      viewPoints.append(viewCenter + viewKeepout*array([cos(45*(pi/180)), 0, sin(45*(pi/180))]))

    upChoice = array([0.9,0.1,0])
    upChoice = upChoice / norm(upChoice)

    viewPoses = []
    for point in viewPoints:
      viewPoses.append(self.GetSensorPoseGivenUp(point, viewCenter, upChoice))

    return viewPoses
    
  def GetSensorPoseGivenUp(self, sensorPosition, targetPosition, upAxis):
    '''Generates the sensor pose with the LOS pointing to a target position and the "up" close to a
       specified up.
    - Input sensorPosition: 3-element desired position of sensor placement.
    - Input targetPosition: 3-element position of object required to view.
    - Input upAxis: The direction the sensor up should be close to.
    - Returns T: 4x4 numpy array (transformation matrix) representing desired pose of end effector
      in the base frame.
    '''
  
    v = targetPosition - sensorPosition
    v = v / norm(v)
  
    u = upAxis - dot(upAxis, v) * v
    u = u / norm(u)
  
    t = cross(u, v)
  
    T = eye(4)
    T[0:3,0] = t
    T[0:3,1] = u
    T[0:3,2] = v
    T[0:3,3] = sensorPosition
  
    return T
    
  def GetTableHeight(self):
    '''Accessor method for the table height -- the z position of the top of the table surface.'''
    
    return self.tableHeight

  def MoveHandToHoldingPose(self):
    '''Moves the hand to a special, pre-designated holding area.'''

    T = eye(4)
    T[0:3, 0] = array([-1, 0,  0])
    T[0:3, 1] = array([ 0, 1,  0])
    T[0:3, 2] = array([ 0, 0, -1])
    T[0:3, 3] = array([-1, 0,  0.30])

    self.MoveHandToPose(T)

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
    
  def MoveSensorToPose(self, T):
    '''Moves the hand of the robot to the specified pose.'''

    self.robot.SetTransform(dot(T, self.sTh))
    self.env.UpdatePublishedBodies()

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
        
  def StartSensor(self):
    '''Starts the sensor in openrave, displaying yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOn)

  def StopSensor(self):
    '''Disables the sensor in openrave, removing the yellow haze.'''

    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
    self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)

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