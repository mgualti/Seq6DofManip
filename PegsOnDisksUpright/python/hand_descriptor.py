'''Provides a class for representing a hand pose and a hand volume.'''

# python
from time import time
from copy import copy
# scipy
from matplotlib import pyplot
from numpy.linalg import inv, norm
from numpy.random import rand, randn
from numpy import arange, arccos, arctan, arctan2, array, ascontiguousarray, ceil, concatenate, \
  cross, dot, eye, linspace, maximum, meshgrid, ones, pi, reshape, round, sqrt, stack, zeros
# self

class HandDescriptor():

  def __init__(self, T, imP, imD, imW):
    '''Creates a HandDescriptor object with everything needed.'''

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

  def GenerateHeightmap(self, env):
    '''Generates a heightmap for the current descriptor given an environment with shape primitives.
    - Input env: rl_environment_pegs_on_disks object.
    - Returns self.image: The image generated for this descriptor.
    '''

    # precomputed values
    dxy = self.imW / self.imP
    cornerXi = (self.center[0] / dxy) - ((self.imP - 1.0) / 2.0)
    cornerYi = (self.center[1] / dxy) - ((self.imP - 1.0) / 2.0)
    value = 0.50 + ((env.tablePosition[2] + env.tableExtents[2]) - self.center[2]) / self.imD
    value = min(max(0.0, value), 1.0)
    self.image = value * ones((self.imP, self.imP, 1), dtype='float32')

    # for each object, compute relevant image indices
    objects = env.objects + env.supportObjects
    for obj in objects:
      objCenter = obj.GetTransform()[0:3, 3]
      oXiLo = max(int(round((objCenter[0] - obj.radius) / dxy - cornerXi)), 0)
      oXiHi = min(int(round((objCenter[0] + obj.radius) / dxy - cornerXi))+1, self.imP)
      if oXiLo >= self.imP or oXiHi < 0: continue
      oYiLo = max(int(round((objCenter[1] - obj.radius) / dxy - cornerYi)), 0)
      oYiHi = min(int(round((objCenter[1] + obj.radius) / dxy - cornerYi))+1, self.imP)
      if oYiLo >= self.imP or oYiHi < 0: continue
      value = 0.50 + ((objCenter[2] + obj.height/2.0) - self.center[2]) / self.imD
      value = min(max(0.0, value), 1.0)
      self.image[oXiLo:oXiHi, oYiLo:oYiHi, 0] = \
        maximum(value, self.image[oXiLo:oXiHi, oYiLo:oYiHi, 0])
    
    return self.image

  def PlotImage(self):
    '''Plots the image descriptor for this grasp.'''

    if self.image is None:
      return

    fig = pyplot.figure()
    pyplot.imshow(self.image[:, :, 0], vmin=0.00, vmax=1.00, cmap='gray')

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