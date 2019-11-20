'''RL agent implementing hierarchical spatial attention (HSA).'''

# python
import os
import pickle
# scipy
from numpy.random import rand, randint
from numpy import array, delete, log2, meshgrid, ravel_multi_index, reshape, unravel_index, zeros
# drawing
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
# self

# AGENT ============================================================================================

class RlAgentHsa:

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    # parameters
    self.n = params["nObjects"]
    self.m = params["worldSize"]
    self.tMax = params["tMax"]
    self.initQ = params["initQ"]    
    self.alpha = params["alpha"]
    self.gamma = params["gamma"]
    self.plotImages = params["plotImages"]
    
    # determine parameters
    self.worldSize = (self.m, self.m, self.m)
    self.endEffectorIdx = self.m**3
    self.observationSize = (2, 2, 2)
    self.L = int(log2(self.m))
    
    # some input checking
    if self.m < 2:
      raise Exception("Because the observation is 2x2x2, the smallest supported world size is 2.")
    if log2(self.m) != self.L:
      raise Exception("The current implementation only supports grid sizes that are powers of 2.")

    # initialize q-function
    # Q = (G1_0, ..., G1_8, G2_0, ..., G2_8, h, l, t, a)
    self.Q = {None:0.0}
    
  def GetQTableSize(self):
    '''Returns the number of Q-values stored in the lookup table.'''
    
    return len(self.Q)

  def GetAction(self, a, l, i):
    '''Gets the next underlying action coordinate (x, y, z) given the current underlying action a,
      level l and abstract action i.
    - Input a: Underlying action coordinate (x, y, z).
    - Input l: The current level in 0, ..., L - 1.
    - Input i: The abstract action index.
    - Returns aa: The next underlying action.
    '''
    
    d = self.m / 2 ** (l+1)
    iCoord = unravel_index(i, self.observationSize)
    return (a[0] + d * iCoord[0], a[1] + d * iCoord[1], a[2] + d * iCoord[2])

  def GetActionsAndObservations(self, s, epsilon):
    '''Gets a list of actions, abstract actions, and observations for each level in the sense sequence.
    - Input s: The current state = (pegs, holes, time).
    - Input epsilon: Takes a random action with probability epsilon.
    - Returns a: The underlying action (world coordinates flattened index).
    - Returns i: The abstract actions, one for each sense level, each of which is an index
      indicating which cell in the observation was selected.
    - Returns o: The  observations, one for each sense level l, each of which is the tuple from
      GetObservation.
    '''
    
    # reached terminal state
    if s is None:
      return None, None, None
    
    # decompose state information
    pegs = s[0:self.n]; disks = s[self.n:2*self.n]; t = s[-1]
    
    # ignore peg-disks
    ignorePegs = []; ignoreDisks = []; i = 0; j = 0
    while i < self.n and j < self.n:
      if pegs[i] == disks[j]:
        ignorePegs.append(i)
        ignoreDisks.append(j)
        i += 1; j += 1
      elif pegs[i] < disks[j]: i += 1
      else: j += 1
    unplacedPegs = delete(pegs, ignorePegs)
    unoccupiedDisks = delete(disks, ignoreDisks)
    
    # compute coordinates for pegs and disks
    pegCoords = []; diskCoords = []; h = False
    for i in xrange(len(unplacedPegs)):
      if unplacedPegs[i] == self.endEffectorIdx:
        h = True
      else:
        pegCoords.append(unravel_index(unplacedPegs[i], self.worldSize))
      diskCoords.append(unravel_index(unoccupiedDisks[i], self.worldSize))    
    
    # initialize outputs
    a = (0, 0, 0); aPrev = None; i = []; o = []

    # take best action, breaking ties randomly
    for l in xrange(self.L):
      observation = self.GetObservation(pegCoords, diskCoords, h, l, t, a)
      if rand() < epsilon:
        abstractAction = randint(8)
        idx = observation + (abstractAction,)
        if idx not in self.Q: self.Q[idx] = self.initQ[t]
      else:
        bestValue = -float('inf'); abstractActions = None
        for abstractAction in xrange(8):
          idx = observation + (abstractAction,)
          if idx not in self.Q: self.Q[idx] = self.initQ[t]
          if self.Q[idx] > bestValue:
            bestValue = self.Q[idx]
            abstractActions = [abstractAction]
          elif self.Q[idx] == bestValue:
            abstractActions.append(abstractAction)
        # break ties randomly
        abstractAction = abstractActions[randint(len(abstractActions))]
      # compute new sensor location and append observation and abstract action to list
      aPrev = a
      a = self.GetAction(a, l, abstractAction)
      o.append(observation); i.append(abstractAction)
      
      # visualization
      if self.plotImages:
        print("Best value: {}".format(bestValue))
        self.PlotImages(pegs, disks, t, l, h, a, aPrev)

    return ravel_multi_index(a, self.worldSize), i, o

  def GetObservation(self, pegs, disks, h, l, t, a):
    '''Gets an HVS observation given the underlying state and current underlying point of focus/action.
    - Input pegs: List of unplaced pegs in global coordinates (x, y, z), excluding any held peg.
    - Input disks: List of unplaced disks in global coordinates (x, y, z).
    - Input h: True if a peg is in the end effector and False otherwise.
    - Input l: The current level.
    - Input t: The current overt time step.
    - Input a: The current point of focus/action (x, y, z).
    - Returns o: The current observation (g1_0, ..., g1_8, g2_0, ..., g2_8, h, l, t).
    '''
    
    # cell size for this level
    d = self.m / 2 ** (l+1)
    
    # initialize grids to empty
    G1 = zeros(self.observationSize, dtype='bool')
    G2 = zeros(self.observationSize, dtype='bool')
    
    # check each partition for a peg/disk
    
    for i in xrange(2):
      for j in xrange(2):
        for k in xrange(2):
      
          x0 = a[0] + i * d; x1 = x0 + d
          y0 = a[1] + j * d; y1 = y0 + d
          z0 = a[2] + k * d; z1 = z0 + d
          
          for peg in pegs:
            if peg[0] >= x0 and peg[0] < x1 and \
               peg[1] >= y0 and peg[1] < y1 and \
               peg[2] >= z0 and peg[2] < z1:
                 G1[i, j, k] = True
                 break
               
          for disk in disks:
            if disk[0] >= x0 and disk[0] < x1 and \
               disk[1] >= y0 and disk[1] < y1 and \
               disk[2] >= z0 and disk[2] < z1:
                 G2[i, j, k] = True
                 break
    
    return tuple(G1.flatten()) + tuple(G2.flatten()) + (h, l, t)
    
  def LoadQFunction(self):
    '''Loads a previously saved dictionary of Q-values.'''
    
    path = os.getcwd() + "/" + "model-hvs.pkl"
    self.Q = pickle.load(open(path, "rb"))
    print("Loaded {}.".format(path))
  
  def PlotCube(self, ax, xMinMax, yMinMax, zMinMax, color, alpha):
    '''https://codereview.stackexchange.com/questions/155585/plotting-a-rectangular-prism'''
    
    xx, yy = meshgrid(xMinMax, yMinMax)
    ax.plot_wireframe(xx, yy, reshape(zMinMax[0], (1, 1)), color=color)
    ax.plot_surface(xx, yy, reshape(zMinMax[0], (1, 1)), color=color, alpha=alpha)
    ax.plot_wireframe(xx, yy, reshape(zMinMax[1], (1, 1)), color=color)
    ax.plot_surface(xx, yy, reshape(zMinMax[1], (1, 1)), color=color, alpha=alpha)
    
    yy, zz = meshgrid(yMinMax, zMinMax)
    ax.plot_wireframe(xMinMax[0], yy, zz, color=color)
    ax.plot_surface(xMinMax[0], yy, zz, color=color, alpha=alpha)
    ax.plot_wireframe(xMinMax[1], yy, zz, color=color)
    ax.plot_surface(xMinMax[1], yy, zz, color=color, alpha=alpha)

    xx, zz = meshgrid(xMinMax, zMinMax)
    ax.plot_wireframe(xx, yMinMax[0], zz, color=color)
    ax.plot_surface(xx, yMinMax[0], zz, color=color, alpha=alpha)
    ax.plot_wireframe(xx, yMinMax[1], zz, color=color)
    ax.plot_surface(xx, yMinMax[1], zz, color=color, alpha=alpha) 
  
  def PlotImages(self, pegs, disks, t, l, h, a, aPrev):
    '''Visualizes a current situation.
    - Input pegs: List of all pegs as a flat coordinate.
    - Input disks: List of all disks as a flat coordinate.
    - Input t: The overt time step.
    - Input l: The current level.
    - Input h: The holding bit.
    - Input a: The underlying action coordinate.
    - Input aPrev: The previous action coordinate.
    - Returns None. An image is shown and the thread is blocked until it is closed.
    '''
    
    # setup plot
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')    
    
    # draw grid linse
    dxyz = 1.0 / self.m
    for i in xrange(self.m + 1):
      for j in xrange(self.m + 1):
        for k in xrange(3):
          x = [i * dxyz]*2
          y = [j * dxyz]*2
          z = [0, 1]
          if k == 0:            
            ax.plot(x, y, z, 'k', alpha=0.25)
          elif k == 1:
            ax.plot(z, x, y, 'k', alpha=0.25)
          else:
            ax.plot(y, z, x, 'k', alpha=0.25)
            
    # draw objects 
    for disk in disks:
      coord = array(unravel_index(disk, self.worldSize)) / float(self.m) + dxyz / 2.0
      ax.scatter(coord[0], coord[1], coord[2], c='b', s=100, marker='o')
    for peg in pegs:
      if peg == self.endEffectorIdx: continue
      coord = array(unravel_index(peg, self.worldSize)) / float(self.m) + dxyz / 2.0
      ax.scatter(coord[0], coord[1], coord[2], c='r', s=100, marker='^')
      
    # draw observation area
    corner = array(aPrev) / float(self.m)
    size = 1.0 / 2.0**l
    self.PlotCube(ax, [corner[0], corner[0] + size], [corner[1], corner[1] + size],
      [corner[2], corner[2] + size], 'y', 0.2)
    
    # draw area selected by robot
    corner = array(a) / float(self.m)
    size = 1.0 / 2.0**(l+1)
    self.PlotCube(ax, [corner[0], corner[0] + size], [corner[1], corner[1] + size],
      [corner[2], corner[2] + size], 'g', 0.2)
    
    # plot properties
    ax.set_xlim3d(0.12, 0.88)
    ax.set_ylim3d(0.12, 0.88)
    ax.set_zlim3d(0.12, 0.88)
    ax.view_init(elev=15, azim=-10)
    ax._axis3don = False
    ax.set_aspect('equal')
    fig.suptitle("t={}. l={}. h={}.".format(t, l, h))
    pyplot.show(block=True)
    
  def SaveQFunction(self):
    '''Saves the current Q-value dictionary to a Python pickle file, model-hvs.pkl.'''
    
    path = os.getcwd() + "/" + "model-hvs.pkl"
    pickle.dump(self.Q, open(path, "wb"))
    print("Saved {}.".format(path))

  def UpdateQFunction(self, o, i, r, oo, ii):
    '''Updates the current q-estimates, according to the Sarsa update rule, given an (overt) time
      step of experience.
    - Input o: A list of observations, [o_0, ..., o_L-1].
    - Input i: A list of abstract actions, [i_0, ..., i_L-1].
    - Input r: The (scalar) reward received after taking this action from this state.
    - Input oo: List of observations in the next (overt) time step, [oo_0, ..., oo_L-1].
    - Input ii: List of abstract actions taken in the next (overt) time step, [oo_0, ..., oo_L-1].
    - Returns None.
    '''

    for l in xrange(self.L):

      idx = o[l] + (i[l],)

      if l != self.L - 1:
        ll = l + 1
        jdx = o[ll] + (i[ll],)
        rr = 0
      else:
        ll = 0
        jdx = None if oo is None else oo[ll] + (ii[ll],)
        rr = r
      
      # update Q
      self.Q[idx] = (1.0 - self.alpha) * self.Q[idx] + self.alpha * (rr + self.gamma * self.Q[jdx])