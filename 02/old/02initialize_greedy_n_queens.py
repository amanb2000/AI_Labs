import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)
    greedy_init = greedy_init.astype(int)

    for col in range(1,N): # Column number
        conflicts = np.zeros(N) # num conflicts in each row.
        for row in range(N):
            conflicts[row] = num_row_conflicts(greedy_init, row, col, N)

        greedy_init[col] = np.random.choice(np.flatnonzero(conflicts == conflicts.min()))


    ### YOUR CODE GOES HERE

    return greedy_init.astype(int)


def num_row_conflicts(map, row, column, N):
    flat = np.zeros(N, int) + row
    up = np.arange(N)
    diff = row-up[column]
    up += diff
    down = np.flip(np.arange(N))
    diff = row-down[column]
    down += diff

    conf = 0

    conf += np.sum( (map==down)[:column] )
    conf += np.sum( (map==up)[:column] )
    conf += np.sum( (map==flat)[:column] )
    return conf




if __name__ == '__main__':
    # You can test your code here 
    # num_row_conflicts(None, 3, 7, 15)
    print(initialize_greedy_n_queens(1000))
