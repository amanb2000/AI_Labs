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


def policy_evaluation_smart(policy, agent, env):
    n = len(env.states)
    P = np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            P[i,j] = env.transition_model[env.states[i], env.states[j], policy[env.states[i]] ]

    P = P*agent.gamma

    print("Policy: ",policy)

    print("P: ",P)
    
    r = np.asarray(env.rewards)
     
    inv_term = np.linalg.inv(P-np.eye(n)) 

    print("Inv_term: ", inv_term)

    u = np.linalg.inv(P-np.eye(n)) @ (-r)

    return u


def policy_evaluation(policy, agent, env):
    U_i_1 = np.zeros(agent.utility.shape)
    
    for s in env.states:
        sum_term = 0
        for s_p in env.states:
            sum_term += env.transition_model[s, s_p, policy[s]] * agent.utility[s_p]

        U_i_1[s] = env.rewards[s] + agent.gamma * sum_term

    return U_i_1



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

    policy = np.random.randint(len(env.actions), size=(len(env.states)))
    agent.utility = np.zeros([len(env.states)])

    ## START: Student code
    unchanged = False
    while not unchanged:
        agent.utility = policy_evaluation(policy, agent, env)
        unchanged = True

        for s in env.states:
            max_sum = None # max_a sum_{s'} P(s' | s, a) * U[s']
            max_a = None
            for a in env.actions:
                pot_max_sum = 0
                for s_p in env.states:
                    pot_max_sum += env.transition_model[s, s_p, a]*agent.utility[s_p]
                if max_sum == None or pot_max_sum > max_sum:
                    max_sum = pot_max_sum
                    max_a = a

            reg_sum = 0
            for s_p in env.states:
                reg_sum += env.transition_model[s, s_p, policy[s]] * agent.utility[s_p]

            if max_sum > reg_sum:
                policy[s] = max_a
                unchanged = False

    ## END: Student code

    return policy
