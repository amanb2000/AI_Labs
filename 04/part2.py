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


def policy_evaluation(policy, agent, env):
    """
    Simplified policy evaluation formula from AIMA.
    """
    U_i_1 = np.zeros(agent.utility.shape) # New utility value for each state at time i+1
    
    for s in env.states: # Iterating through states. 
        sum_term = 0 # Sum term for the expected future reward from entering state s.
        for s_p in env.states: # iterating through second-degree states. 
            sum_term += env.transition_model[s, s_p, policy[s]] * agent.utility[s_p] # adding the expected value from the second-degree state s_p (s prime)

        U_i_1[s] = env.rewards[s] + agent.gamma * sum_term # Adding reward and a discounted future reward to obtain the utility value for states s

    return U_i_1 # Returning utility values. 



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
    # np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states))) # Initializing the policy as a 1D array (will be reshaped by the end)
    agent.utility = np.zeros([len(env.states)]) # initializing the utility as a 1D array (will be reshaped by the end). 

    ## START: Student code
    unchanged = False # Flag for whether the policy has changed at all. 
    while not unchanged:
        agent.utility = policy_evaluation(policy, agent, env) # Evaluating policy
        unchanged = True # Unsetting the unchanged flag. 

        for s in env.states: # iterating through each state
            max_sum = None # max_a sum_{s'} P(s' | s, a) * U[s']
            max_a = None # argument that maximizes the sum. 
            for a in env.actions: # Calculating the maximum sum over all possible actions. 
                pot_max_sum = 0
                for s_p in env.states:
                    pot_max_sum += env.transition_model[s, s_p, a]*agent.utility[s_p]
                if max_sum == None or pot_max_sum > max_sum:
                    max_sum = pot_max_sum
                    max_a = a

            reg_sum = 0 # calculating the non-action-optimized sum.
            for s_p in env.states:
                reg_sum += env.transition_model[s, s_p, policy[s]] * agent.utility[s_p] # Expected value from the action based on the policy.

            if max_sum > reg_sum: # if the action-optimized sum is greater than the policy-based sum, we need to continue revising the policy.
                policy[s] = max_a # Update the action for state 
                unchanged = False # set the unchanged flag. 

    ## END: Student code

    return policy
