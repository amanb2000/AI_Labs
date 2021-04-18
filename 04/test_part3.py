
# test_part3.py  (adopted from the work of Anson Wong)
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
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from mab_util import random_MAB_env, run_experiment
from part3 import MAB_agent

MAX_RUNTIME_PE = 0.0001 #s

def test1(create_graph=False) -> int:
    """
        Experiment with your agent and choose the best algorithm 
        to maximize reward in the MAB environment
        - The code below spawns a random MAB environment and 
          lets the agent policy run on it
        - The reward from each episode in each experiment is 
          stored and plots are created to evaluate the agent
    """
    # Settings
    num_arms = np.random.randint(50,100) # bandit arm probabilities of success
    num_experiments = 500 # number of experiments to perform
    num_eps = 400 # number of steps (episodes)
    output_dir = os.path.join(os.getcwd(), "output")
    env = random_MAB_env(num_arms) # initialize arm probabilities
    points = 0

    print("\n\t\tTEST 1")

    # Run time experiment
    init_time = time.time()
    actions, rewards = run_experiment(env, MAB_agent(num_arms), num_eps)
    run_time = (time.time() - init_time)/num_eps
    print("Each epsiode runs in {0: .6f} seconds".format(run_time))
    if run_time > MAX_RUNTIME_PE:
        print("Your episode runtime exceeds the threshold!!")
        return points
    points += 1

    # Run multi-armed bandit experiments
    print("Running multi-armed bandits with number of actions = {}".format(num_arms))
    R = np.zeros((num_eps,))  # reward history sum
    A = np.zeros((num_eps, num_arms))  # action history sum
    for i in range(num_experiments):
        # Run an experiment with a new agent but on the same environment
        actions, rewards = run_experiment(env, MAB_agent(num_arms), num_eps)
        # # Logging
        # if (i + 1) % (num_experiments / 100) == 0:
        #     print("[Experiment {}/{}] ".format(i + 1, num_experiments) +
        #         "n_steps = {}, ".format(num_eps) +
        #         "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
        R += rewards
        for j, a in enumerate(actions): A[j][a] += 1
    R_avg =  R / np.float(num_experiments)
    probs = env.get_probs()

    # Plot avg reward vs step count and save figure
    if create_graph:
        plt.plot(R_avg/max(probs), ".")
        plt.xlabel("Step")
        plt.ylabel("% of max reward")
        plt.grid()
        ax = plt.gca()
        plt.xlim([1, num_eps])
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, "rewards.png"), bbox_inches="tight")
        plt.close()

    """
    Scoring Table
            |    0    |    1    |    2    |    3    |
    --------------------------------------------------
    50      |   <70%  | 70-80%  | 80-87%  |  >87%   |
    75      |   <70%  | 70-87%  | 87-93%  |  >93%   |
    100/200 |   <80%  | 80-90%  | 90-95%  |  >95%   |
    """
    # 50 episodes
    n = 50
    r = np.mean(R_avg[n-5:n])/max(probs) # average over 5 reward values
    print("After {} episodes, % max reward: {}".format(n, r))
    if r < 0.7: points += 0
    elif r < 0.8: points += 1
    elif r < 0.87: points += 2
    else: points += 3

    # 75 episodes
    n = 75
    r = np.mean(R_avg[n-5:n])/max(probs) # average over 5 reward values
    print("After {} episodes, % max reward: {}".format(n, r))
    if r < 0.70: points += 0
    elif r < 0.87: points += 1
    elif r < 0.93: points += 2
    else: points += 3
     
    # 100/200 episodes
    n = 100
    r_100 = np.mean(R_avg[n-5:n])/max(probs) # average over 5 reward values
    print("After {} episodes, % max reward: {}".format(n, r_100))
    n = 200
    r_200 = np.mean(R_avg[n-5:n])/max(probs) # average over 5 reward values
    print("After {} episodes, % max reward: {}".format(n, r_200))
    if r_100 < 0.8 or r_200 < 0.8: points += 0
    elif r_100 < 0.9 or r_200 < 0.9: points += 1
    elif r_100 < 0.95 or r_200 < 0.95: points += 2
    else: points += 3

    return points

if __name__ == '__main__':
    print("You have scored {}/10 in part 1".format(test1(create_graph=False)))
    
