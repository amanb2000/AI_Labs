# part2.py: Project 4 Part 2 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

###
# Imports
###

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), 1))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code

    ## END: Student code

    return policy