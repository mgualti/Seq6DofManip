'''An HSA agent where Q-values are updated according to the Monte Carlo rule.'''

# python
# scipy
# drawing
# self
from rl_agent_hsa import RlAgentHsa

# AGENT ============================================================================================

class RlAgentHsaMc(RlAgentHsa):

  def __init__(self, params):
    '''Initializes the agent with an optimistic policy.
    - Input params: System parameters data structure.
    '''

    RlAgentHsa.__init__(self, params)
    
    self.Sums = {}
    self.Counts = {}    

  def UpdateQFunction(self, o, i, r):
    '''Updates the current q-estimates given an episode of experiences.
    - Input o: A list of observation lists, [[o_0, ..., o_L-1]_0, ... [o_l0, ..., o_L-1]_T].
    - Input i: A list of abstract action lists, formatted as above.
    - Input r: A list of rewards, one for each time step, [r_1, ..., r_T].
    - Returns None.
    '''
    
    for t in xrange(self.tMax):
      
      for l in xrange(self.L):
  
        idx = o[t][l] + (i[t][l],)
        
        # discounted sum of rewards from now onward
        Return = 0
        for j in xrange(t, self.tMax):
          Return += self.gamma**(j - t) * r[j]
        
        # update Q
        if idx not in self.Sums:
          self.Sums[idx] = 0.00
          self.Counts[idx] = 0.00
        self.Sums[idx] += Return
        self.Counts[idx] += 1.00
        self.Q[idx] = self.Sums[idx] / self.Counts[idx]