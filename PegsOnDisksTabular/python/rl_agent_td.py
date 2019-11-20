'''RL agnet implementing temporal difference (TD) algorithms operating on the underlying state.'''

# python
# scipy
from numpy.random import randint
# self

# AGENT ============================================================================================

class RlAgentTd:

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    # parameters
    self.n = params["nObjects"]
    self.m = params["worldSize"]
    self.initQ = params["initQ"]    
    self.alpha = params["alpha"]
    self.gamma = params["gamma"]
    
    # determine number of actions
    self.nActions = self.m**3

    # initialize q-function
    # Q = (p_1, ... p_n, d_1, ... d_n, t, a)
    self.Q = {None:0.0}

  def GetAction(self, s):
    '''Finds the highest-valued action given the current state.
    - Input s: The current state.
    - Returns action: The highest-valued action according to the current q-estimate.
    '''
    
    # reached terminal state
    if s is None: return None
    
    # decompose state information    
    t = s[-1]    
    
    # take best action, breaking ties randomly
    bestValue = -float('inf'); actions = None
    for action in xrange(self.nActions):
      idx = s + (action, )
      if idx not in self.Q: self.Q[idx] = self.initQ[t]
      if self.Q[idx] > bestValue:
        bestValue = self.Q[idx]
        actions = [action]
      elif self.Q[idx] == bestValue:
        actions.append(action)
    
    # break ties randomly
    return actions[randint(len(actions))]
    
  def GetQTableSize(self):
    '''Returns the number of Q-values stored in the lookup table.'''
    
    return len(self.Q)

  def UpdateQFunction(self, s, a, r, ss, aa):
    '''Updates the current q-estimates, according to the Sarsa update rule, given a time step of
      experience.
    - Input s: The current state.
    - Input a: The current action.
    - Input r: The reward received after being in s and taking a.
    - Input ss: The next state.
    - Input aa: The next action actually taken.
    - Returns None.
    '''
    
    idx = s + (a, ); jdx = None if ss is None else ss + (aa, )
    self.Q[idx] = (1.0 - self.alpha) * self.Q[idx] + self.alpha * (r + self.gamma * self.Q[jdx])