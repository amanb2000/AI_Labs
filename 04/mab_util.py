# mab_util.py: Project 4
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

import numpy as np
import copy
import time

# Time based seed
np.random.seed(int(time.time()))

class random_MAB_env:
    """
        This class defines a random multi-arm bandit environment with N arms 
        similar to the one shown in Reinforcement Learning: An Introduction
        http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms
        self.__probs = [np.random.uniform() for i in range(num_arms)]

    def init_probability(self, probs):
        self.__num_arms = len(probs)
        self.__probs = copy.deepcopy(probs)

    def size(self) -> int:
        """
            Size of the Multi-Armed Bandit problem is defined as the number
            of arms which can be pulled to receive a reward
        """
        return self.__num_arms

    def get_arms(self) -> list:
        """
            Getter function that returns a list of all available arms
        """
        return [i for i in range(self.__num_arms)]

    def pull(self, arm: int) -> float:
        """
            Environment step function that returns a reward for the pulled arm
        """

        if 0 > arm or  arm >= self.__num_arms:
            print("ERROR: ARM {} NOT RECOGNISED".format(arm))
            return 0.0

        # Pull arm and get stochastic reward (1 for success, 0 for failure)
        return 1.0 if (np.random.uniform()  < self.__probs[arm]) else 0.0

    def get_probs(self):
        """
            Get a copy of current bandit probabilities
        """
        ## IMPLEMENTATION
        return copy.deepcopy(self.__probs)


# Multi-Armed Bandit Experiment function
def run_experiment(env: 'random_MAB_env', agent: 'MAB_agent', num_eps: int) -> (np.array, np.array):
    """
        Run a finite number of episodes of a MAB agent on an MAB environment

        Return
        ---------
        actions, rewards
            arrays containing the actions and rewards associated with each episode
            size = (num_eps)
    """
    actions, rewards = [], []
    for _ in range(num_eps):
        arm = agent.get_action() # sample policy
        reward = env.pull(arm) # take step + get reward
        agent.update_state(arm, reward) # update agent state
        actions.append(arm)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)
