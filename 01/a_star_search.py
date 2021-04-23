import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """

    # Extracting init_node and goal node for convenience.
    init_state = problem.init_state
    goal = problem.goal_states[0]
    # Instantiating number of nodes expanded and frontier size at 0 
    num_nodes_expanded = 0
    max_frontier_size = 0


    # PART I: Create frontier priority queue, set `explored`, dictionary denoting parents.
    Q = queue.PriorityQueue() # Tracks priority for expanding nodes in the form (f(n), n) where f(n) = h(n) + g(n) 
    frontier = set() # frontier tracks the same items as Q. Must be synchronized. Used for its ability to quickly check if an item has been seen via hashing.
    explored = set() # Set of explored nodes.
    parent = {} # Function for tracking both the parent of each node and the g(n) function (i.e. cost to get from the initial node to the node in question).

    # Adding the initial node to the data structures.
    Q.put( (problem.manhattan_heuristic(init_state, goal), init_state) ) 
    frontier.add(init_state)
    parent[init_state] = (None, 0) # NOTE: The cost-to-get is for the object i in parent[i]

    # Part II: While the frontier is not empty:
        # pop a node v from the frontier
        # test if it's the GOAL => break out if it is
        # add the node to the `explored` set.
        # for each of v's potential actions v_i
            # If v_i isn't in `explored` or `frontier`: insert into the frontier
            # If child IS in frontier BUT has a higher cost:
                # Replace the child's parent with v
                # Update the child's cost

    # Initializing v::int and v_get::tuple to None so they are available outside of the While loop scope.
    v = None
    v_get = None
    while(not Q.empty()):
        # Checking frontier size
        max_frontier_size = max(max_frontier_size, len(frontier)) # Updating based on the max. 
        num_nodes_expanded+=1 # Incrementing the number of nodes expanded.
        
        # Extracting the next lowest-cost node from the priority queue 
        v_get = Q.get()
        v = v_get[1] # Integer representing the node.
        vfn = v_get[0] # Integer representing the estimated cost f(n) = g(n) + h(n) of the node.

        # Removing the node from the frontier.
        try:
            frontier.remove(v)
        except:
            pass

        # Checking if the goal condition has been met by v.
        if(v == goal):
            break
        
        # Updating the explored set to include v.
        explored.add(v)

        # Processing all of the 'children' of v.
        for (garbage, vi) in problem.get_actions(v):
            if vi not in explored and vi not in frontier:
                vi_gn = parent[v][1]+1 # cost-to-go for v plus 1
                parent[vi] = (v, vi_gn) # setting the parent and CTG info for vi

                # f(n) = g(n) + h(n)
                vi_fn = vi_gn + problem.manhattan_heuristic(vi, goal)
                
                # Adding vi to the priority queue and to the frontier (must remain synchronized).
                Q.put( (vi_fn, vi) )
                frontier.add(vi)

            elif vi in frontier and parent[vi][1] > parent[v][1]+1:
                parent[vi] = (v, parent[v][1]+1)
                vi_fn = parent[vi][1] + problem.manhattan_heuristic(vi, goal)

                Q.put( (vi_fn, vi) )
                frontier.add(vi)

    # Part III: If `goal` has no parent in the dictionary, return []. Otherwise, trace path through the graph.

    # Returning the unsolvable 
    if v != goal:
        # print("UNSOLVABLE")
        return [], 0, 0
        # return [init_state, goal], 0, 0 # Debugging statement...

    path = []
    cur_node = goal

    # Constructing the path...
    while cur_node != None:
        path.append(cur_node)
        cur_node = parent[cur_node][0]
    
    path.reverse()

    # Returning the path and the nodes expanded, max frontier size.
    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    print("Nodes Expanded: \t{}".format(num_nodes_expanded))
    print("Max Frontier: \t\t{}".format(max_frontier_size))
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)
    # Experiment and compare with BFS    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    print("Nodes Expanded: \t{}".format(num_nodes_expanded))
    print("Max Frontier: \t\t{}".format(max_frontier_size))
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)
    # Experiment and compare with BFS
