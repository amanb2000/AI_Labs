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
    pass


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
    easy_solution = sudoku_solver(easy_input_board)  # Solve
    conflicts = sudoku_conflicts(easy_solution, free_variables)  # Check the number of conflicts
