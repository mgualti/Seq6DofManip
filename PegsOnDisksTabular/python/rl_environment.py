'''Reinforcement learning (RL) environment for the tabular pegs on disks domain.'''

# python
# scipy
from numpy.random import choice
from numpy import arange, delete

class RlEnvironment:

  def __init__(self, params):
    '''Initializes a random environment of the specified size.'''

    # parameters
    self.n = params["nObjects"]
    self.m = params["worldSize"]
    self.tMax = params["tMax"]
    self.endEffectorIdx = self.m**3
    self.worldSize = (self.m, self.m, self.m)
    self.rewardShaping = params["rewardShaping"]

    # create pegs and disks (order of execution matters)
    self.GenerateRandomDisks()
    self.GenerateRandomPegs()

    # set simulator time
    self.t = 0

  def GenerateRandomDisks(self):
    '''Places disks uniformly at random in the grid. Disks cannot be on top of each other.'''

    gridIdxs = arange(self.m**3)
    self.disks = list(choice(gridIdxs, self.n, replace=False))

  def GenerateRandomPegs(self):
    '''Places pegs uniformly at random in the grid. Pegs cannot be on top of each other, and
      initially, we do not want then to be on top of disks or in the end effector.'''

    gridIdxs = arange(self.m**3)
    # make sure no peg is above a disk
    gridIdxs = delete(gridIdxs, self.disks)
    # make sure a peg is not above another peg
    self.pegs = list(choice(gridIdxs, self.n, replace=False))

  def GetState(self):
    '''Returns the current state of the environement.
    - Returns s: Tuple (pegs, disks, time). Can assume the peg/disk indices are sorted.
    '''
    
    if self.t > self.tMax - 1: return None
    return tuple(sorted(self.pegs) + sorted(self.disks) + [self.t])
    
  def IsPick(self):
    '''Returns True if the action from this state is a pick action and False otherwise.'''
    
    return self.endEffectorIdx not in self.pegs
    
  def IsPlace(self):
    '''Returns True if the action from this state is a place action and False otherwise.'''
    
    return self.endEffectorIdx in self.pegs

  def Transition(self, a):
    '''Moves the state of the environment forward by one time step.
    - Input a: The desired action, (x, y, z).
    - Returns r: The reward received for taking action a in the current state.
    '''

    r = 0.0  
    
    pegInEndEffector = self.endEffectorIdx in self.pegs
    if pegInEndEffector: # attempt place
      abovePeg = a in self.pegs
      if not abovePeg: # can place only if a peg is not at the action location                
        heldPegIdx = self.pegs.index(self.endEffectorIdx)   
        # check if placing correctly
        atDisk = a in self.disks
        if atDisk: r = 1.0
        # perform transition
        self.pegs[heldPegIdx] = a
    
    else: # attempt pick
      # are there any pegs at the action?
      atPeg = a in self.pegs
      if atPeg:
        pegIdx = self.pegs.index(a)
        # removing a placed peg?
        if a in self.disks: r = -1.0
        elif self.rewardShaping: r = 1.0
        # perform transition
        self.pegs[pegIdx] = self.endEffectorIdx
    
    # return result
    self.t = self.t + 1
    return r