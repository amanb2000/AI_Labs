# part1_2.py: Project 4 Part 1 script
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
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states) # Initializing the policy as a 1-D array (will be reshaped at the end).
    agent.utility = np.zeros(len(env.states)) # Initializing the utility as a 1-D array (will be reshaped at the end). 


    ## START: Student code
    # NOTE: The `env` hasa `transition_model`. 
    # transition_model:  Matrix of size (SxSxA) specifying all of the
    #                    transition probabilities
    
    U_p = np.zeros(agent.utility.shape) # U', U prime from the equations.

    delta = 0 # Maximum change in utility of any state in an iteration.

    cnt = 0 # counter for the number of iterations.
    while cnt < max_iter:
        delta = 0 # resetting delta to zero for the iteration.
        agent.utility = U_p.copy() # Copying out the value of U prime obtained in the previous iteration. 

        for s in env.states: # Looping through each possible state
            max_term = None # max_{a\in A} (\sum_{s'} P(s'|s,a) U[s'] )
            for a in env.actions: # loop to find the maximizing a value. 
                pot_max = 0 # potential maximum
                for s_p in env.states: # s_p is s', we now sum over P(s'|s,a)*U[s']
                    pot_max += env.transition_model[s,s_p,a]*agent.utility[s_p] # adding the expected value of s'*p(s') 
                if max_term == None or pot_max > max_term: # updating the max_term
                    max_term = pot_max
            
            U_p[s] = env.rewards[s] + agent.gamma * max_term # U' = R[s] + gamma * max{...}

        if delta < np.max(np.abs(agent.utility - U_p)): # Updating the value of delta based on the most changed utility value. 
            delta = np.max(np.abs(agent.utility - U_p))

        if delta < eps*(1-agent.gamma)/agent.gamma: # Ending the loop of delta is below the cutoff value. 
            break

    # Generating the policy: For each state, we take the action with the maximum expected value.
    for s in env.states:
        max_a = None
        max_EV = None

        for a in env.actions: # Iterating over each possible actions to maximize
            pot_max_EV = 0 
            for s_p in env.states: # Iterating over each s', second degree states
                pot_max_EV += env.transition_model[s,s_p,a]*agent.utility[s_p] # determining the expected value of the action based on the second-degree states. 
            if max_EV == None or pot_max_EV > max_EV: # updating the maximum expected value. 
                max_EV = pot_max_EV
                max_a = a
        
        policy[s] = max_a # Updating the policy based on the maximizing action. 

    agent.utility = agent.utility.reshape( [len(agent.utility), 1] ) # re-shaping the utility to fit with the assignment requirements (TIP: use the `@` matrix multiplication operator instead of using weirdly shaped column vectors).

    ## END Student code
    return policy
