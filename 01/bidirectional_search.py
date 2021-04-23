from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """

    # STEP I: Create dictionaries and sets for init-starting and goal-starting trees.

    # Extracting initial node and goal node for convenience.
    init_node = problem.init_state
    goal = problem.goal_states[0]

    # Setting up data structures to track progress.
    I_set = set() # Set that functions as a hash table to quickly check if 
                  # the frontiers intersect.
    I_set.add(init_node)
    I_dict = {} # Dictionary for storing the parent of each object added to set.
    I_dict[init_node] = None
    I = deque() # Queue for storing the vertices that still need to be processed.
    I.append(init_node) 

    # Same data structures for branching out from goal node.
    G_set = set()
    G_set.add(goal)
    G_dict = {}
    G_dict[goal] = None
    G = deque()
    G.append(goal)

    # STEP II: While I, G aren't empty, we alternate between expanding each frontier.
    v = None 
    while len(G) != 0 and len(I) != 0:
        v = I.popleft() # Getting the next element in the QUEUE for the init wave

        I_set.add(v) # Adding the vertex to the set of processed init vertices.

        if v in G_set: # Checking if we can terminate
            break

        actions = problem.get_actions(v) # Extracting adjacent members 

        for(v1, v2) in actions: # Iterating through actions
            if v2 not in I_set: # If the node has not been visited...
                I_dict[v2] = v # Add to the dictionary with the parent `v`
                I_set.add(v2) # Add to the set for quick hash-based search.
                I.append(v2) # Add to the queue.


        # Same procedure is followed for the other wavefront.
        v = G.popleft()
        G_set.add(v)

        if v in I_set:
            break

        actions = problem.get_actions(v)

        for(v1, v2) in actions:
            if v2 not in G_set:
                G_dict[v2] = v
                G_set.add(v2)
                G.append(v2)

    # STEP III: Construct the path from the initial node to the goal, if it exists.


    if goal not in G_set and goal not in I_set:
        return [], None, None

    # Construct the path from "meet in the middle" node v -> init_node
    path = [v]
    cur_node = I_dict[v] #parent of v on the init side
    while cur_node != None:
        path.append(cur_node)
        cur_node = I_dict[cur_node]

    path.reverse() # Reverse the path so it traverses init_node->v

    # Add nodes from v -> goal node to the existing path.
    cur_node = G_dict[v]
    while cur_node != None:
        path.append(cur_node)
        cur_node = G_dict[cur_node] 


    # Return values. 
    max_frontier_size = 0
    num_nodes_expanded = 0
    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    # path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)


    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!
