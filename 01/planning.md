# Planning

__Personal planning documentation__


## Framing

* Uninformed search on explicit graph.
* Implement `A*` on 2D maze.
	- "Best first search" (weighted graph)
	- Selects next node $n$ path that minimizes $f(n) = g(n) + h(n)$
		- $g(n)$ is the cost of path to node $n$
		- $h(n)$ is the heuristic for the cost to get from $n$ to the `goal`.
* Submission via [Autolab](https://q.utoronto.ca/courses/205162/discussion_topics/1025231).
## Particular Tasks

* `breadth_first_search.py` (pretty straight forward?)
* `bidirectional_search.py` 
	- Bidirectional (???) BFS algorithm.
	- Searches simultaneously from initial + goal states until it "meets in the middle"
* `a_star_search.py` 
	- SUPPLIED with a heuristic function!

### Additions/Constraints

* Need to make additional tests :/ 
* Tests and visualizations should be in separate files (no additional libraries allowed).
* *Return `None` or `[]` when no solutions can be found* (all searches).
* `search_problems.py` has classes and a function that's needed for solutions.
	- `node: {parent, state, action, path_cost}` for state space search.
	- Abstract classes `SearchProblem, SimpleSearchProblem`: Main problem classes inherit from.
		- `GraphSearchProblem` class --> `breadth_dirst_search` and `bidirectional_search`.
		- `GridSearchProblem` class --> `a_star_search`.
	* `get_random_grid_problem()`: Produces random `GridSearchProblem` instances.
* Allowable libraries:
	- Python standard library.
* Don't need to submit `search_problems.py`.
* Can assume `problem.goal_states[1]` (`[0]`?) to access goal state since there is *only one goal state*. 
* 

### Grading Criterion

- Commenting: Should be able to understand algorithms from JUST reading comments.
- 300 second limit on the execution time. 


## Uninformed Search

### Part I: Breadth-First Search
- Input: Undirected graph $G = (V, E)$
- Vertices $v$ labeled with unique integers.
- Uniformly weighted edges -> weight = # edges to get to target.
- `GridSearchProblem` constructor:
	- Initial state
	- List of goal states (only 1).
	- Graph G = (V,E)
- [ ] Implement BFS in `breadth_first_search` in `breadth_first_search.py`.
	- Tuple with list of states representing path from init state -> goal state
	- Number of nodes expanded (keep a count).
	- Maximum frontier size during the search.
	- (ONLY TUPLE WITH LIST OF STATES WILL BE GRADED).
	- Allowed to implement own datastructures.
	- `deque` structure is recommended.
	- `set()` may also be useful.

Deque Datastructure:
- `collections.deque`
















