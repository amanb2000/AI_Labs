
# part3_solution.py  (adopted from the work of Anson Wong)
#
# --
# Artificial Intelligence
# ROB 311 Winter 2021
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Teaching Assistant:
# Sepehr Samavi
# sepehr@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

"""
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
"""
import numpy as np

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it 
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms #private
        ## IMPLEMENTATION

        # Hyperparams:
        self.Q_init = 1./num_arms # initialization values for each Q

        # Initializations:
        self.N_a = np.zeros(num_arms) # 1D array to count the number of times each action has been taken. 
        self.Q = np.ones(num_arms) * self.Q_init # Expected value of each action (running average)
        self.t = 1 # time step number
        self.c = .1 # tuning parameter for how much we value reduction in uncertainty.

    def update_state(self, action, reward):
        """
            TODO:
            Based on your choice of algorithm, use the the current action and 
            reward to update the state of the agent. 
            Optinal function, only use if needed.
        """
        ## IMPLEMENTATION
        self.N_a[action] += 1 # incrementing number of times the action has been taken
        self.t += 1 # incrementing time
        self.Q[action] = self.Q[action] + (1/self.N_a[action])*(reward-self.Q[action]) # Applying incremental update to expected value. 

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        EVs = self.Q + self.c * np.sqrt(np.log(self.t)/(self.N_a+.1)) # expected value is the Q-value plus the sqrt-log measure of uncertainty (AIMA).
        return np.argmax(EVs) # Returning the argument that maximizes expected value.



