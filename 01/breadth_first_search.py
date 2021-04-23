from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes
    instances of SimpleSearchProblem (or its derived classes) 
    and provides a valid and optimal path from the initial state 
    to the goal state. Useful for testing your bidirectional and 
    A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Extracting initial node and goal node for convenience.
    init_node = problem.init_state
    goal = problem.goal_states[0]

    # "Actions" are just tuples of integers. Integers represent vertices.
    # The only way to get information on a vertex is to call `problem.get_action(vertex)`.
    
    # Step I: Create Dictionary of Nodes and deque() datastructures.

    node_dict = {} # Primary purpose of dictionary is to track parent values.
    node_dict[problem.init_state] = Node(None, True, None, 0) # Populating dictionary with only known node: initial state.

    Q = deque() # Queue gives an order to explore new nodes.
    Q.append(init_node) # Adding initial node to the Queue 

    # Step II: While the Q isn't empty...
        # `popleft()` the leftmost element
        # Check if it's the goal. If so, break out.
        # Process children into `node_dict` if they aren't in it yet.
            # if `vertex not in node_dict`, add it in with the parent just popped.

    while len(Q) != 0:
        v = Q.popleft() # Processing the next element in the queue.

        if v == goal: # Checking if we've reached the goal!
            break

        # Processing the children of v.
        actions = problem.get_actions(v)

        for (v1, v2) in actions: # Iterating through each potential action.
            if v2 not in node_dict: # Adding the child node to the queue if it's not already been processed.
                node_dict[v2] = Node(v, True, None, node_dict[v].path_cost+1)
                Q.append(v2)
 

    # Step III: Construct the path from the first to last (or return empty list if the goal node is not in the dictionary key list.

    if goal not in node_dict: # Did not find goal node. Return empty list.
        return [], None, None

    # Creating the path to the goal based on the parent dictionary.
    path = [] 
    path.append(goal)
    cur_node = node_dict[goal].parent
    while cur_node != None:
        path.append(cur_node)
        cur_node = node_dict[cur_node].parent

    path.reverse()
            
    

    # Returning results!
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
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    print("\nAttempting Stanford Large Dataset...")

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
