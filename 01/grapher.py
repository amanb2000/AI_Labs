"""
Graphing Script: 
1. Satisfiability:
    - x: p_occ
    - y: number of runs satisfiable
2. Nodes generated:
    - x: p_occ
    - y: Average number of nodes generated

Parameters:
    Fixed: n_runs = 100; 
    N in (20, 100, 500) // two above plots for each
        p_occ in range(0.1, 0.05, 0.9)

Recorded Results:
    - Satisfiability?
    - Number of nodes generated?

To-Do Items:
- [x] Set up `a_star_search` function call on `get_random_grid_problem`.
- [x] Set up pandas array record keeping.
- [x] Set up pandas array record saving.
- [x] Set up the experiment loop(s) and variables/ranges.
- [x] Set up parallization loop.
- [x] Test on small subset, ensure proper output.
- [x] Run full experiment.

"""

# IMPORTS
import queue
import numpy as np
import pandas as pd
from a_star_search import a_star_search as astra
from search_problems import Node, GridSearchProblem, get_random_grid_problem
from tqdm import tqdm

import concurrent.futures

def run_experiment():

    df = get_df() 
    n_runs = 100
    p_occs = np.linspace(0.1, 0.9, 17)
    ddict = {}

    pbar = tqdm(total=3*len(p_occs)*100)

    exp_num = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        jobs = []
        prams = []

        for N in (20, 100, 500): 
            for p_occ in p_occs:
                for run in range(n_runs):
                    problem = get_random_grid_problem(p_occ, N, N)
                    # Solve it
                    # path, num_nodes_expanded, max_frontier_size = astra(problem)
                    f = executor.submit(astra, problem)
                    jobs.append(f)
                    prams.append((p_occ, N, run))


        for i in range(len(jobs)):
            path, num_nodes_expanded, max_frontier_size = jobs[i].result()
            (p_occ, N, run) = prams[i]
            d = {'run': exp_num, 'N': N, 'p_occ': round(p_occ,2), 'nodes_generated': num_nodes_expanded, 'SAT': path != []}
            df = df.append(d, ignore_index=True)
            pbar.update(1)
            exp_num+=1

    pbar.close()


    df.to_csv('experiment_results.csv')

def get_df():
    d = {'run': [], 'N': [], 'p_occ': [], 'nodes_generated': [], 'SAT': []}
    df = pd.DataFrame(data=d)
    # df = df.append({'run': 0, 'N': 100, 'p_occ': 0.25, 'nodes_generated': 100, 'SAT': 0}, ignore_index=True)
    # df.to_csv('experiment_results.csv')

    return df


# astra tests
def test_astra():
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = astra(problem)
    print("Nodes Expanded: \t{}".format(num_nodes_expanded))
    print("Max Frontier: \t\t{}".format(max_frontier_size))
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)
    # Experiment and compare with BFS




def main():
    run_experiment()

if __name__ == "__main__":
    main()


