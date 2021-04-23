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
    greedy_init = np.zeros(N, dtype=int) # Initializing our greedy initialization with zeros.
    greedy_init[0] = np.random.randint(0, N) # Randomly placing the first column's queen.
    greedy_init = greedy_init.astype(int) # Ensuring it's of type `int`

    # Each queen has a star-shaped damage area. 
    # We use 3 arrays to store them as we move from left to right.
    # The horizontal damage from a queen is stored in `horizontals`. 
    # The diagonal-upward damage from a queen in stored in `diag_ups`.
    # The diagonal-downward damage from a queen is stored in `diag_downs`.
    horizontals = np.zeros(N, dtype=int) 
    diag_ups = np.zeros(N, dtype=int)
    diag_downs = np.zeros(N, dtype=int)

    # We begin by adding the first queen's damage area to the array.
    horizontals[greedy_init[0]] = 1
    diag_ups[greedy_init[0]] = 1
    diag_downs[greedy_init[0]] = 1

    for col in range(1,N): # Column number
        # We "roll" the diagonals as we traverse the columns so that they accurately 
        # depict the way that diagonal damage propagates diagonally upward and downward.
        diag_ups = np.roll(diag_ups, -1)
        diag_ups[-1] = 0 # Since roll functions like `>>` in C, we need to set the value that rolled from beginning <--> end to 0.
        diag_downs = np.roll(diag_downs, 1)
        diag_downs[0] = 0 # Since roll functions like `>>` in C, we need to set the value that rolled from beginning <--> end to 0.

        # `conflicts` represents the total conflicts resulting from horizontal, diagonal up, and diagonal down damage
        # from queens to the left of this column. 
        conflicts = horizontals + diag_ups + diag_downs
        # Our row decision is a random uniform choice among the minimum conflict indexes
        row_decision = np.random.choice(np.flatnonzero(conflicts == conflicts.min()))
        greedy_init[col] = row_decision # Updating our greedy initialization

        # Adding 1 to our damage collectors at our row decision (will be rolled up/down at the beginning of next iteration)
        diag_ups[row_decision] += 1
        diag_downs[row_decision] += 1
        horizontals[row_decision] += 1

    # After our last column is assigned a row, we return.
    return greedy_init.astype(int)




if __name__ == '__main__':
    # You can test your code here 
    # num_row_conflicts(None, 3, 7, 15)
    print(initialize_greedy_n_queens(1000))
