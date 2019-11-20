'''An HSA agent where Q-values are updated according to the Q-learning rule.'''

# python
# scipy
# drawing
# self
from rl_agent_hsa import RlAgentHsa

# AGENT ============================================================================================

class RlAgentHsaQl(RlAgentHsa):

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    RlAgentHsa.__init__(self, params)

  def UpdateQFunction(self, o, i, r, oo):
    '''Updates the current q-estimates given an (overt) time step of experiences.
    - Input o: A list of observations, [o_0, ..., o_L-1].
    - Input i: A list of abstract actions, [i_0, ..., i_L-1].
    - Input r: The (scalar) reward received after taking this action from this state.
    - Input oo: List of observations in the next (overt) time step, [oo_0, ..., oo_L-1].
    - Returns None.
    '''
    
    for l in xrange(self.L):

      idx = o[l] + (i[l],)

      if l != self.L - 1:
        ll = l + 1
        nextObservation = o[ll]
        rr = 0
      else:
        ll = 0
        nextObservation = None if oo is None else oo[ll]
        rr = r
      
      if nextObservation is not None:
        # take max over next actions
        tt = nextObservation[-1]
        bestValue = -float('inf'); nextAction = None
        for ii in xrange(8):
          jdx = nextObservation + (ii,)
          if jdx not in self.Q: self.Q[jdx] = self.initQ[tt]
          if self.Q[jdx] > bestValue:
            bestValue = self.Q[jdx]
            nextAction = ii
        jdx = nextObservation + (nextAction, )
      else:
        # terminal state
        jdx = None
        
      # update Q
      self.Q[idx] = (1.0 - self.alpha) * self.Q[idx] + self.alpha * (rr + self.gamma * self.Q[jdx])