'''Provides a class for representing a hand pose and a hand volume.'''

# python
from time import time
from copy import copy
# scipy
from matplotlib import pyplot
from numpy.linalg import inv, norm
from numpy.random import rand, randn
from scipy.ndimage.morphology import grey_dilation
from numpy import arange, arccos, arctan, arctan2, array, ascontiguousarray, ceil, concatenate, \
  cross, dot, eye, linspace, maximum, meshgrid, ones, pi, reshape, round, sqrt, stack, zeros
# self
import point_cloud
from c_extensions import SetImageToMaxValues

class HandDescriptor():

  def __init__(self, T, imP, imD, imW):
    '''Creates a HandDescriptor object with everything needed.
    - Input T: Pose (4x4 homogeneous transform matrix) of the hand in the base/world frame.
    - Input imP: Resolution of the square heightmap to be generated.
    - Input imD: Size of image, in meters, of the descriptor volume in the hand depth direction.
    - Input imW: Size of image, in meters, of the descriptor volume in the hand closing and axis
      directions.
    '''

    self.T = T

    # hand size (used for drawing)
    self.depth = 0.075
    self.width = 0.085
    self.height = 0.01

    # image size (used for image descriptor)
    self.imP = imP
    self.imD = imD
    self.imW = imW

    # hand axes
    self.axis = T[0:3, 0]
    self.binormal = T[0:3, 1]
    self.approach = -T[0:3, 2]
    self.center = T[0:3, 3]

    self.bottom = self.center - 0.5 * self.depth * self.approach
    self.top = self.center + 0.5 * self.depth * self.approach

    # internal variables
    self.image = None
    
  def ComputeHeightmap(self, X, tableHeight):
    '''Computes a heightmap representing the hand volume. Views hand contents down the approach vector.
    - Input X: Point cloud (nx3 numpy array) in hand reference frame.
    - Input tableHeight: Height of the table, assumed to be normal to z axis.
    - Returns im: The heightmap, normalized to [0, 1], representing points in the descriptor volume.
    '''
    
    # initialize image with table height
    # (assumes the grasp is normal to the table)
    tableHeightInHandFrame = tableHeight - self.center[2]
    value = (tableHeightInHandFrame + (self.imD / 2.0)) / self.imD
    value = min(max(0.0, value), 1.0)
    im = value * ones((self.imP, self.imP), dtype='float32')

    # get image coordinates of each point
    coordsX = (X[:, 0] + (self.imW / 2.0)) * ((self.imP - 1) / self.imW)
    coordsY = (X[:, 1] + (self.imW / 2.0)) * ((self.imP - 1) / self.imW)
    coords = stack((coordsX, coordsY), axis=1)
    coords[coords < 0] = 0
    coords[coords > (self.imP-1)] = (self.imP-1)

    # get height of each point
    values = (X[:, 2] + (self.imD / 2.0)) / self.imD

    # set image values at each coordinate
    coords = ascontiguousarray(coords, dtype='float32')
    values = ascontiguousarray(values, dtype='float32')
    SetImageToMaxValues(im, im.shape[0], im.shape[1], coords, values, values.shape[0])

    return im

  def GenerateHeightmap(self, cloud, tableHeight):
    '''Generates image representing contents of descriptor volume.
    - Input cloud: Cloud of the entire scene (besides the table), in the base/world refernece frame.
    - Input tableHeight: Location of the top of the table surface in the z direction. (Assumes table
      is normal to z axis and objects are above the table in the +z direction.)
    '''

    X = self.GetHandPoints(cloud)
    self.image = self.ComputeHeightmap(X, tableHeight)
    self.image = grey_dilation(self.image, size=3)
    self.image = self.image.reshape((self.image.shape[0], self.image.shape[1], 1))
    return self.image
    
  def GetHandPoints(self, cloud):
    '''Determines which points are in the descriptor volume.
    - Input cloud: Point cloud in the base/world frame.
    - Returns X: Point cloud in the hand frame, including only points in the descriptor volume.
    '''

    workspace = [(-self.imW/2, self.imW/2), (-self.imW/2, self.imW/2), (-self.imD/2, self.imD/2)]

    X = point_cloud.Transform(inv(self.T), cloud)
    return point_cloud.FilterWorkspace(workspace, X)

  def PlotImage(self):
    '''Plots the descriptor image using Matplotlib.'''

    if self.image is None:
      return

    fig = pyplot.figure()
    pyplot.imshow(self.image[:, :, 0], vmin=0.00, vmax=1.00, cmap='gray', interpolation="none")

    pyplot.title("Hand Image")
    fig.axes[0].set_xticks([])
    fig.axes[0].set_yticks([])

    pyplot.show(block=True)

# UTILITIES ========================================================================================

def PoseFromApproachAxisCenter(approach, axis, center):
  '''Given grasp approach and axis unit vectors, and center, get homogeneous transform for grasp.'''

  T = eye(4)
  T[0:3, 0] = axis
  T[0:3, 1] = cross(approach, axis)
  T[0:3, 2] = approach
  T[0:3, 3] = center

  return T