import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha=0.07#0.08
        self.gamma=0.8#0.8

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done,eps):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if not done:
                next_action = self.epsilon_greedy(self.Q, next_state, self.nA , eps) # epsilon-greedy action
                self.Q[state][action] = self.update_Q_learning(self.alpha, self.gamma, self.Q, \
                                                  state, action, reward, next_state, next_action)
                #self.Q[state][action] = self.update_expected_sarsa(self.alpha,self.nA,eps, self.gamma, self.Q, \
                #                                  state, action, reward, next_state, next_action)
        if done:
                self.Q[state][action] = self.update_Q_learning(self.alpha, self.gamma, self.Q, \
                             state, action, reward)
                #self.Q[state][action] = self.update_expected_sarsa(self.alpha,self.nA,eps, self.gamma, self.Q, \
                #                                  state, action, reward)
          
    def epsilon_greedy(self,Q, state, nA, eps):
        """Selects epsilon-greedy action for supplied state.
    
        Params
        ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
        """
        if np.random.random() > eps: # select greedy action with probability epsilon
            return np.argmax(Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)#np.random.choice(np.arange(env.action_space.n))


    def update_Q_learning(self,alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        maxq=np.max(Q[next_state])
        
        # get value of state, action pair at next time step
        #Qsa_next = Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (gamma *maxq)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value
        return new_value
    def update_expected_sarsa(self,alpha,nA,eps, gamma, Q, state, action, reward, next_state=None, next_action=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
     
        policy_s = np.ones(nA) * eps / nA  # current policy (for next state S')
        policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA) # greedy action
        qvalue = np.dot(Q[next_state], policy_s)         # get value of state at next time step
    
        # get value of state, action pair at next time step
        #Qsa_next = Q[next_state][next_action] if next_state is not None else 0    
        target = reward + (gamma *qvalue)               # construct TD target
        new_value = current + (alpha * (target - current)) # get updated value
        return new_value
