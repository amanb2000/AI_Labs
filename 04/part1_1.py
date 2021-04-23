# part1_1.py: Project 4 Part 1 script
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
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleaning robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code

    # Terminal states: If s1 = 0 or 5 (terminal states), then s2 = s1 for all actions.
    # We leave these at zero.

    # Shortcuts for 'left action' and 'right action'.
    l_ac = 0
    r_ac = 1

    for i in range(1,5): # i represents the current state
        for j in range(6): # j represents the next state
            if j == i+1: # If the next state is one to the right:
                P[i,j,l_ac] = 0.05 # probability of i -> j given you go left
                P[i,j,r_ac] = 0.8 # probability of i -> j given you go right
            elif j == i-1: # If the next state is one to the left:
                P[i,j,l_ac] = 0.8
                P[i,j,r_ac] = 0.05
            elif i == j: #If the next state is the same as the current state:
                P[i,j,l_ac] = 0.15
                P[i,j,r_ac] = 0.15 
    
    # Error checking for impossible transition probabilities. 
    assert(P[0,1,l_ac] == P[0,1,r_ac] == 0)
    assert(P[5,4,l_ac] == P[5,4,l_ac] == 0)

    ## END: Student code
    return P
