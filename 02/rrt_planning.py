"""
    Problem 3 Template file
"""
import random
import math

import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for a problem setup given by the "RRT_dubins_problem" class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file "rrt_planning.py". Your implementation
   can be tested by running "RRT_dubins_problem.py" (see the "main()" function).
2. Read all class and function documentation in "RRT_dubins_problem.py" carefully.
   There are plenty of helper functions in the class that you should use.
3. Your solution must meet all the conditions specificed below.
4. Below are some DOs and DONTs for this problem.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random points
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out related issues and will be generously set.
2. The planning function must return a list of nodes that represent a collision free path
   from the start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must be a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation of the node to understand the terminology)
3. The returned path should be a valid list of nodes with a Dubins-style path connecting the nodes. 
   i.e. the list should have the start node at index 0 and goal node at index -1. 
   For all other indices i in the list, the parent node for node i should be at index i-1,  
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   "RRT_dubins_problem.map_area"

DO(s) and DONT(s)
-------------------
1. DO rename the file to rrt_planning.py for submission.
2. Do NOT change change the "planning" function signature.
3. Do NOT import anything other than what is already imported in this file.
4. We encourage you to write helper functions in this file in order to reduce code repetition
   but these functions can only be used inside the "planning" function.
   (since only the planning function will be imported)
"""

def planning(rrt_dubins, display_map=False):
    """
        Execute RRT planning using dubins-style paths. Make sure to populate the node_lis

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    # Fix Randon Number Generator seed
    random.seed(1)

    # LOOP for max iterations
    i = 0
    final = None
    while i < rrt_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw)
         # rrt_dubins.map_area = [min_x, max_x, min_y, max_y] is where we sample, (x, y) from.
         # yaw is somewhere between 0 and 2pi radians.
         # state should 90-99% of the time be random. Otherwise, it should be the `objective` node.
        x = random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1])
        y = random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1])
        yaw = random.uniform(0, math.pi)

        goal = False

        min_i = 0
        # Find an existing node nearest to the random vehicle state
        if(random.random() < 0.9): # 90% chance of taking the random exploration path
            # Find the nearest node to this random node.
            goal = rrt_dubins.Node(x,y,yaw) # `goal` is a randomly instantiate node.
            min_dist = (rrt_dubins.node_list[0].x-x)**2 + (rrt_dubins.node_list[0].y-y)**2 # initializing minimum distance for node search.
            min_i = 0 
            for i in range(len(rrt_dubins.node_list)): 
                # Scanning over all existing nodes to find the nearest one to our exploratory goal node.
                dist = (rrt_dubins.node_list[i].x-x)**2 + (rrt_dubins.node_list[i].y-y)**2
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
            
            # new_node = rrt_dubins.propogate(rrt_dubins.node_list[min_i], rrt_dubins.node(x, y, yaw)) #example of usage
        else: # Non-exploration path
            # Find the nearest node to the objective.
            goal = rrt_dubins.goal # Our goal node in this case is the
            # Initializing minimum distance 
            min_dist = (rrt_dubins.node_list[0].x-rrt_dubins.goal.x)**2 + (rrt_dubins.node_list[0].y-rrt_dubins.goal.y)**2
            min_i = 0
            for i in range(len(rrt_dubins.node_list)):
                # Scanning over all existing nodes to find the nearest one to our exploratory goal node.
                dist = (rrt_dubins.node_list[i].x-rrt_dubins.goal.x)**2 + (rrt_dubins.node_list[i].y-rrt_dubins.goal.y)**2
                if dist < min_dist:
                    min_dist = dist
                    min_i = i

        # Generating a new node using propagate that connects to our closest existing node to our goal node from above.
        new_node = rrt_dubins.propogate(rrt_dubins.node_list[min_i], goal)

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        node_ok = False # boolean for if the node was accepted or rejected by the collision subroutine
        if rrt_dubins.check_collision(new_node):
            node_ok = True
            rrt_dubins.node_list.append(new_node) # Storing all valid nodes

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_dubins.draw_graph()

        # Check if new_node is close to goal AND is a valid non-collision node.
        if new_node.is_state_identical(rrt_dubins.goal) and node_ok: 
            print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list)) # print number of iterations + number of nodes.
            final = new_node # `final` will be the last node in our path to the goal since it is `identical` to the goal node.
            break

    if i == rrt_dubins.max_iter: # If we reached max iterations, we print it.
        print('reached max iterations')

    # Return path, which is a list of nodes leading to the goal
    path = []
    while(final != None): # Keep following the parent chain until we reach `None`, the first node's parent. 
       path.append(final)
       final = final.parent
    path.reverse() # Reversing the path so it's a valid dubin's path from the start to the objective.

    return path # Returning path.
