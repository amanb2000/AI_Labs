import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization) # Extracting N for easy usage.
    solution = initialization.copy() # Making a copy of the initialization for our solution.
    num_steps = 0
    max_steps = 1000
    found = False # Variable for if we actually found a solution OR if we ran out of time.

    for idx in range(max_steps):
        # 1: iterate through all columns, add column numbers that conflict to a list.
        conflict_columns = [] # List of columns that conflict
        num_steps+=1 

        for col in range(N): # iterating through all columns,
            if num_conflicts(solution, solution[col], col, N) != 0:
                conflict_columns.append(col)
                
        # 2: if the list is empty, return.
        if len(conflict_columns) == 0:
            found = True # no conflicts! Solution has been found!
            break 

        # 3: randomly select a conflicting column from the list.
        focus_column = np.random.choice(np.asarray(conflict_columns))

        # 4: find the minimum conflict ROW for that column's queen (see greedy solution). Set the row value.
        solution[focus_column] = greedy_reassignment(solution, focus_column, N)


    if found: # if we did find a solution, we return it.
        return solution, num_steps
    else: # else if we didn't find the solution, we return empty list and -1 steps.
        return [], -1

def num_conflicts(map, row, column, N):
    """
    Returns number of conflicts for the given [row, column] within `map`.
    """

    # I use masks for all possible conflict types (horizontal, up diagonal, down diagonal)
    # If the current solution[i] == mask[i], that means that there's one conflict in [row, column]
    flat = np.zeros(N, int) + row # Mask for horizontal damage
    up = np.arange(N) # mask for diagonally upward damage
    diff = row-up[column] # correcting offset so the path goes through [row, column]
    up += diff 
    down = np.flip(np.arange(N)) # mask for diagonally downward damage
    diff = row-down[column] # correcting offset so the path goes through [row, column]
    down += diff

    conf = 0 # Counter for number of conflicts of all types.

    conf += np.sum( (map==down)[:column] ) # Diagonal down conflicts from columns to the left.
    conf += np.sum( (map==up)[:column] ) # Diagonal upward conflicts from columns to the left.
    conf += np.sum( (map==flat)[:column] ) # Horizontal conflicts from columns to the left.

    conf += np.sum( (map==down)[column+1:] ) # Diagonal down conflicts from columns to the right.
    conf += np.sum( (map==up)[column+1:] ) # Diagonal up conflicts from columns to the right.
    conf += np.sum( (map==flat)[column+1:] ) # Horizontal conflicts from columns to the right.
    return conf

def greedy_reassignment(map, col, N):
    """
    Function to select a random greedy reassignment for the queen in column `col` of map `map`.
    """
    conflicts = np.zeros(N) # num conflicts in each row.
    for row in range(N): # calling upon the `num_conflicts` function to efficiently calculate total conflicts of each possible row assignment.
        conflicts[row] = num_conflicts(map, row, col, N)

    return np.random.choice(np.flatnonzero(conflicts == conflicts.min())) # Randomly uniformly selecting a row from the minimum conflict rows.

if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1054
    # Use this after implementing initialize_greedy_n_queens.py
    print("starting init")
    assignment_initial = initialize_greedy_n_queens(N)
    print("done init")
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    print("Starting min_conflict")
    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    print("Ending min_conflict")
    # Plot the solution produced by your algorithm

    plot_n_queens_solution(assignment_solved)
