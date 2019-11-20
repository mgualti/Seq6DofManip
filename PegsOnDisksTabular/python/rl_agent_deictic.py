'''RL agent implementing Deictic Image Mapping.'''

# python
import os
import pickle
# scipy
from numpy.random import rand, randint
# drawing
# self

# AGENT ============================================================================================

class RlAgentDeictic:

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

    # initialize q-function
    # Q = (h, t, g1, g2)
    self.Q = {None:0.0}
    
  def GetAbstractAction(self, a, pegs, disks):
    '''Generates abstract action, as seen by the agent.
    - Input a: An integer indicating the location on the m^3 grid.
    - Input pegs: List of pegs in the grid.
    - Input disks: List of disks in the grid.
    - Returns: The abstract action, indicating if the selected cell has a peg, a disk, or both.
    '''
    
    return (a in pegs, a in disks)
  
  def GetQTableSize(self):
    '''Returns the number of Q-values stored in the lookup table.'''
    
    return len(self.Q)

  def GetActionAndObservation(self, s, epsilon):
    '''Given the current state, computes an observation and selects an abstract action.
    - Input s: The current state = (pegs, holes, time).
    - Input epsilon: Takes a random action with probability epsilon.
    - Returns a: The underlying action (world coordinates flattened index).
    - Returns i: The abstract action, (g1, g2), bits indicating the presense/absence of a peg/disk
      at the underlying action location.
    - Returns o: The  observation, (h, t), a bit indicating if something is in the hand and the
      current  time.
    '''
    
    # reached terminal state
    if s is None: return None, None, None
    
    # decompose state information
    pegs = s[0:self.n]; disks = s[self.n:2*self.n]; t = s[-1]
    
    # determine observation
    h = self.endEffectorIdx in pegs
    o = (t, h)

    if rand() < epsilon:
      # sample an action uniformly at random
      a = randint(self.m**3)
      i = self.GetAbstractAction(a, pegs, disks)
      idx = o + i
      if idx not in self.Q: self.Q[idx] = self.initQ[t]
    else:
      # take best action, breaking ties randomly
      bestValue = -float('inf'); aEqual = None; iEqual = None
      for a in xrange(self.m**3):
        i = self.GetAbstractAction(a, pegs, disks)
        idx = o + i
        if idx not in self.Q: self.Q[idx] = self.initQ[t]       
        if self.Q[idx] > bestValue:
          bestValue = self.Q[idx]
          iEqual = [i]
          aEqual = [a]
        elif self.Q[idx] == bestValue:
          iEqual.append(i)
          aEqual.append(a)
      # break ties randomly
      iaIdx = randint(len(iEqual))
      i = iEqual[iaIdx]
      a = aEqual[iaIdx]

    return a, i, o
  
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
    - Input o: An observation, (h, t).
    - Input i: An abstract action, (g1, g2).
    - Input r: The (scalar) reward received after taking this action from this state.
    - Input oo: An observation in the next time step.
    - Input ii: An abstract action in the next time step.
    - Returns None.
    '''

    # compute state-action index
    idx = o + i
    jdx = None if oo is None else oo + ii
    
    # update Q
    self.Q[idx] = (1.0 - self.alpha) * self.Q[idx] + self.alpha * (r + self.gamma * self.Q[jdx])