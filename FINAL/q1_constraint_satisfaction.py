# Any additional import statements will be removed by Autolab
import numpy as np

# Permissible values for Sudoku puzzles (excluding 0, which represents a variable cell to be filled in)
VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def count_cell_conflicts(board: np.ndarray, free_variables: np.ndarray, idx: int, jdx: int) -> int:
    """
    Counts the number of row, column, and square conflicts that board[idx, jdx] participates in.

    :param board: input array describing a partially completed Sudoku board (zeros are unassigned variables)
    :param free_variables: boolean array such that free_variables[idx, jdx] is True iff idx, jdx indexes a cell which is
                           not fixed in the input description of the problem.
    :param idx: column index
    :param jdx: row index
    :return: sum of row, column, and square conflicts with the assignment to idx, jdx
    """
    val = board[idx, jdx]
    if val == 0 or not free_variables[idx, jdx]:  # Only assigned free variables can be considered to have conflicts
        return 0
    else:
        row = board[idx, :]
        row_square = 3 * (idx // 3)
        col = board[:, jdx]
        col_square = 3 * (jdx // 3)
        square = board[row_square:row_square + 3, col_square:col_square + 3]
        row_conflicts = np.count_nonzero(row == val) - 1
        col_conflicts = np.count_nonzero(col == val) - 1
        square_conflicts = np.count_nonzero(square == val) - 1
        return row_conflicts + col_conflicts + square_conflicts


def sudoku_conflicts(board: np.ndarray, free_variables: np.ndarray) -> np.ndarray:
    """
    Return a boolean array indicating, for each cell, whether that cell is in a conflict (i.e., constraint violation).
    Unassigned (0) and fixed input values (free_variables[idx, jdx] == False) are always considered conflict-free.

    :param board: input array describing a partially completed Sudoku board (zeros are unassigned variables)
    :param free_variables: boolean array such that free_variables[idx, jdx] is True iff idx, jdx indexes a cell which is
                           not fixed in the input description of the problem.
    :return: array of ints with the number of conflicts each cell participates in: only assigned free variables are
             counted (i.e., the cells containing fixed input values from 1-9)
    """
    n = board.shape[0]
    conflicts = np.zeros(board.shape, dtype=int)
    for idx in range(n):
        for jdx in range(n):
            conflicts[idx, jdx] = count_cell_conflicts(board, free_variables, idx, jdx)
    return conflicts > 0



def deep_copy_domains(domains):
    n_dom = []

    for row in domains:
        tmp = []
        for d in row:
            if type(d) != bool:
                tmp.append(d.copy())
            else:
                tmp.append(False)
        n_dom.append(tmp)

    return n_dom

def select_unassigned(domains):
    """
    Select the maximally constrained entry. Returns False if an error occurs, returns (i,j) for the selected item
    otherwise.
    """
    max_i = None
    max_j = None
    min_len = None

    for i in range(9):
        for j in range(9):
            if type(domains[i][j]) != bool:
                if min_len == None or len(domains[i][j]) < min_len:
                    min_len = len(domains[i][j])
                    max_i = i
                    max_j = j
    
    return max_i, max_j

def domain_update(ds_in, i_s, j_s, asg):
    ds_cpy = deep_copy_domains(ds_in)
    ds_cpy[i_s][j_s] = False

    # Iterate through rows and columns:
    for i in range(9):
        if type(ds_cpy[i][j_s]) != bool and asg in ds_cpy[i][j_s]:
            (ds_cpy[i][j_s]).remove(asg)
        if type(ds_cpy[i_s][i]) != bool and asg in ds_cpy[i_s][i]:
            (ds_cpy[i_s][i]).remove(asg)

    # Iterate through the local box:
    i_st = (i_s // 3)*3
    j_st = (j_s // 3)*3

    for i in range(i_st, i_st+3):
        for j in range(j_st, j_st+3):
            if type(ds_cpy[i][j]) != bool and asg in ds_cpy[i][j]:
                ds_cpy[i][j].remove(asg)

    return ds_cpy



def backtrack(assignment: np.ndarray, domains):
    """
    Backtrack function from AIMA pg. 215. Either returns a solution numpy array or `False` to indicate failure to 
    find solution on the branch in question. 
    """
    
    # Check if the assignment has zeros. If not, check that there are no conflicts and return the result of that check.
    if 0 not in assignment:
        return assignment

    # Select an unassigned variable (using a method `select_unassigned` that minimizes number of conflicts).
    i_s, j_s = select_unassigned(domains)
    
    # Iterate i through the domain of the selected variable.
    for asg in domains[i_s][j_s]:
        # Make a copy of the assignment array to work with for the rest of this...
        ds_cpy = deep_copy_domains(domains)
        ass_cpy = np.copy(assignment)
        # Add i to the assignment (and update the constraint Set array).
        # Update ds_cpy based on assignment
        ds_cpy = domain_update(ds_cpy, i_s, j_s, asg)
        # Update ass_cpy based on assignment
        ass_cpy[i_s,j_s] = asg
        # Recursive call!
        result = backtrack(ass_cpy, ds_cpy)
        if type(result) != bool:
            return result

    return False

def get_allowable_entries(board, i, j):
    i_st = (i//3)*3
    j_st = (j//3)*3

    allowable = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for num in board[i,:]:
        if num in allowable:
            allowable.remove(num)

    for num in board[:,j]:
        if num in allowable:
            allowable.remove(num)

    for nums in board[i_st:(i_st+3),j_st:(j_st+3)]:
        for num in nums:
            if num in allowable:
                allowable.remove(num)

    return allowable

def sudoku_solver(board: np.ndarray) -> np.ndarray:
    """
    Solves a standard 9x9 Sudoku puzzle.
    If you are unable to completely solve a puzzle, Autolab will still reward you for the number of non-conflicting
    assignments, but ONLY if you fill ALL empty (i.e., 0 in board) cells with an integer value from 1-9.

    :param board: 9x9 array of ints containing the Sudoku puzzle. Unknown values, which you must fill in, are 0.
    :return: 2D array with the same non-zero elements as the input board, and the zero-elements filled with integers
    from 1-9 that satisfy row, column, and 3x3 sub-square Sudoku constraints.
    """

    ### STUDENT CODE GOES HERE
    # Use a mix of search and inference for CSPs to solve Sudoku
    
    # Step I: generate the domains set list.
    domains = []

    for i in range(9):
        domain_add = []
        for j in range(9):
            if board[i,j] == 0:
                domain_add.append(get_allowable_entries(board, i, j))
            else:
                domain_add.append(False)
        domains.append(domain_add)

    # Testing domain functions
    result = backtrack(board, domains)

    if type(result) == bool:
        return board
    return result

if __name__ == '__main__':
    # These are 2 of the 3 puzzles that Autolab will grade you on

    # Easy sudoku puzzle from the PDF handout (zeros are values you need to fill in)
    easy_input_board = np.array([[5, 0, 1, 8, 9, 0, 2, 7, 6],
                                 [0, 4, 0, 0, 2, 7, 0, 0, 0],
                                 [8, 0, 7, 1, 0, 3, 0, 0, 0],
                                 [0, 9, 6, 0, 0, 0, 0, 5, 1],
                                 [0, 0, 8, 0, 0, 5, 0, 0, 0],
                                 [0, 5, 0, 9, 1, 6, 4, 8, 0],
                                 [0, 0, 2, 7, 3, 8, 0, 1, 4],
                                 [1, 0, 0, 2, 0, 0, 0, 6, 7],
                                 [0, 0, 0, 0, 5, 0, 0, 2, 0]], dtype=int)

    # Hard sudoku puzzle
    hard_input_board = np.array([[9, 0, 4, 0, 6, 0, 0, 8, 0],
                                 [8, 0, 0, 0, 1, 0, 0, 0, 2],
                                 [1, 0, 0, 9, 0, 0, 0, 7, 0],
                                 [4, 0, 0, 0, 0, 0, 5, 0, 0],
                                 [0, 1, 6, 0, 0, 0, 0, 0, 0],
                                 [7, 0, 0, 0, 3, 1, 0, 0, 0],
                                 [0, 3, 0, 0, 0, 7, 0, 9, 0],
                                 [6, 0, 0, 2, 0, 4, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int)

    # Example of how your code will be called (will throw an error until you complete sudoku_solver above)
    free_variables = easy_input_board == 0
    
    # print("\nstarting easy solution...")
    easy_solution = sudoku_solver(easy_input_board)  # Solve
    # print("done easy solution!")
    # print("\nstarting hard solution...")
    hard_solution = sudoku_solver(hard_input_board)
    # print("done hard solution!")
    # print(hard_solution)
    conflicts = sudoku_conflicts(hard_solution, free_variables)  # Check the number of conflicts
    # print("num conflicts: {}".format(np.sum(conflicts)))
