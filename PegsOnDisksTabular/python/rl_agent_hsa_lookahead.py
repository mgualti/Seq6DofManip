'''RL agent implementing lookahead hierarchical spatial attention (HSA).'''

# python
import os
import pickle
# scipy
from numpy.random import rand, randint
from numpy import delete, floor, log2, ravel_multi_index, unravel_index
# drawing
# self

# AGENT ============================================================================================

class RlAgentHsaLookahead:

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
    self.octGridSize = (2, 2, 2)
    self.endEffectorIdx = self.m**3
    self.L = int(log2(self.m))
    
    # some input checking
    if self.m < 2:
      raise Exception("Because the observation is 2x2x2, the smallest supported world size is 2.")
    if log2(self.m) != floor(log2(self.m)):
      raise Exception("The current implementation only supports grid sizes that are powers of 2.")

    # initialize q-function
    # Q = (h, l, t, g1, g2)
    self.Q = {None:0.0}
    
  def GetQTableSize(self):
    '''Returns the number of Q-values stored in the lookup table.'''
    
    return len(self.Q)

  def GetAbstractAction(self, pegs, disks, l, a, iIdx):
    '''Generates the abstract action, as seen by the agent.
    - Input pegs: List of unplaced pegs (3d world coordinates).
    - Input disks: List of unoccupied disks (3d world coordinates).
    - Input l: The current level, 0, ..., L-1.
    - Input a: The current focus or octant cell corner (3d world coordinates).
    - Input iIdx: The index 0, ..., 7 of the octant to interpret as an abstract action.
    - Returns i: The abstract action as (g1, g2), whether or not an unplaced/unoccupied peg/disk is
      present in the selected octant.
    '''
    
    # cell size for this level
    d = self.m / 2 ** (l+1)
    
    # initialize indicators
    g1 = False; g2 = False
    
    # convet the abstract action index into a 3d coordinate into the octgrid
    iCoord = unravel_index(iIdx, self.octGridSize)
    
    # determine extents of the selected octant
    x0 = a[0] + iCoord[0] * d; x1 = x0 + d
    y0 = a[1] + iCoord[1] * d; y1 = y0 + d
    z0 = a[2] + iCoord[2] * d; z1 = z0 + d
    
    # is a peg in this cell?
    for peg in pegs:
      if peg[0] >= x0 and peg[0] < x1 and \
         peg[1] >= y0 and peg[1] < y1 and \
         peg[2] >= z0 and peg[2] < z1:
           g1 = True
           break
    
    # is a disk in this cell?
    for disk in disks:
      if disk[0] >= x0 and disk[0] < x1 and \
         disk[1] >= y0 and disk[1] < y1 and \
         disk[2] >= z0 and disk[2] < z1:
           g2 = True
           break
    
    # the abstract action representation
    return (g1, g2)
    
  def GetAction(self, a, l, i):
    '''Gets the next underlying action coordinate (x, y, z) given the current underlying action a,
      level l and abstract action i.
    - Input a: Underlying action coordinate (x, y, z).
    - Input l: The current level in 0, ..., L - 1.
    - Input i: The abstract action index.
    - Returns aa: The next underlying action.
    '''
    
    d = self.m / 2 ** (l+1)
    iCoord = unravel_index(i, self.octGridSize)
    return (a[0] + d * iCoord[0], a[1] + d * iCoord[1], a[2] + d * iCoord[2])

  def GetActionsAndObservations(self, s, epsilon):
    '''Gets a list of actions, abstract actions, and observations for each level in the sense sequence.
    - Input s: The current state = (pegs, holes, time).
    - Input epsilon: Takes a random action with probability epsilon.
    - Returns a: The underlying action (world coordinates flattened index).
    - Returns i: The abstract actions, one for each sense level, each of which is the appearance of
      the selected cell, (g1, g2).
    - Returns o: The  observations, one for each sense level l, each of which is (h, l, t).
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
    a = (0, 0, 0); i = []; o = []
    
    for l in xrange(self.L):
      # compute the observation
      observation = (h, l, t)
      if rand() < epsilon:
        # take a random action
        iIdx = randint(8)
        abstractAction = self.GetAbstractAction(pegCoords, diskCoords, l, a, iIdx)
        idx = observation + (abstractAction, )
        if idx not in self.Q: self.Q[idx] = self.initQ[t]
      else:
        # take best action, breaking ties randomly
        bestValue = -float('inf'); iIdxs = None; abstractActions = None
        for iIdx in xrange(8):
          abstractAction = self.GetAbstractAction(pegCoords, diskCoords, l, a, iIdx)
          idx = observation + (abstractAction, )
          if idx not in self.Q: self.Q[idx] = self.initQ[t]
          if self.Q[idx] > bestValue:
            bestValue = self.Q[idx]
            iIdxs = [iIdx]
            abstractActions = [abstractAction]
          elif self.Q[idx] == bestValue:
            iIdxs.append(iIdx)
            abstractActions.append(abstractAction)
        # break ties randomly
        tieBreakIdx = randint(len(abstractActions))
        iIdx = iIdxs[tieBreakIdx]
        abstractAction = abstractActions[tieBreakIdx]
      # compute new sensor location and append observation and abstract action to list
      a = self.GetAction(a, l, iIdx)
      o.append(observation)
      i.append(abstractAction)

    return ravel_multi_index(a, self.worldSize), i, o
    
  def LoadQFunction(self):
    '''Loads a previously saved dictionary of Q-values.'''
    
    path = os.getcwd() + "/" + "model-hvs.pkl"
    self.Q = pickle.load(open(path, "rb"))
    print("Loaded {}.".format(path))
    
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