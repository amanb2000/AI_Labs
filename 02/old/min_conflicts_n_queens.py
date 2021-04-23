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

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000
    found = False # Variable for if we actually found a solution OR if we ran out of time.

    for idx in range(max_steps):
        # 1: iterate through all columns, add column numbers that conflict to a list.
        conflict_columns = set()

        for col in range(N):
            if col in conflict_columns:
                continue

            x1 = solution[col]
            y1 = col
            for pot_conf in range(N):
                if pot_conf == col:
                    continue
                if(pot_conf in conflict_columns):
                    continue

                x2 = solution[pot_conf]
                y2 = pot_conf

                if conflict(x1, y1, x2, y2):
                    conflict_columns.add(col)
                    conflict_columns.add(pot_conf)
                
        # 2: if the list is empty, return.
        if len(conflict_columns) == 0:
            found = True
            break # no conflicts! Solution has been found!

        # 3: randomly select a conflicting column from the list.
        focus_column = np.random.choice(np.asarray(list(conflict_columns)))

        # 4: find the minimum conflict ROW for that column's queen (see greedy solution).
        conflicts = np.zeros(N) # number of conflicts in each row of the focus column.
        for i in range(N): # Iterating through previously assigned queens.
            if i == focus_column: # Skipping the current column in question
                continue

            x1 = solution[i] # Row of queen in column i of the current solution
            y1 = i # column of queen in question from current solution.

            y2 = focus_column # column of potential queen.
            for x2 in range(N):
                conflicts[x2] += conflict(x1, y1, x2, y2) # incrementing the number of conflicts for the row by 1.
        
        solution[focus_column] = np.random.choice(np.flatnonzero(conflicts == conflicts.min()))


        # 5: set the new row value.

        # end for


    if found:
        return solution, num_steps
    else:
        return [], -1

def conflict(x1,y1, x2,y2):
    """
    Helper function to 
    """
    if x1 == x2 or y1 == y2:
        return 1
    if abs(x1-x2) == abs(y1-y2):
        return 1
    return 0

if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1000
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
