# ROB311 Lab 02: Planning

_10% of final grade_.

## Overview

 - Logical inference over definite clauses.
 - Local search to solve `N-queens` constraint SAT problem.
 - Randomized motion planning solver for `Dubins-type vehicle`.

## Part I: Inference with Definite Clauses

__What is a Definite Clause?__ A *disjunction* (OR) of literals where only one literal is NOT negated (e.g., {!a || !b || c}).A

__What is Entailment?__ Whether or not a fact flows from another set of facts logically. I.e., whether or not set A entails proposition/sentence B.


## Part II: N-Queens Problem

### II.1: Greedy Initialization

 - One queen per column.
 - Row numbers in set {0, 1, ..., N-1}
 - Return value is 1 x N vector. ith entry is row # of queen in ith column.

Pseudocode:
 - Instantiate empty 1 x N vector.
 - Add queen at position 0,0.
 - For 1... N:
	- Make an N x 1 vector representing the number of conflicts with the previous columns for each possible column location.
	- Select the argmin for the N x 1 vector, add a queen there.
 - End

```
conflict( (x1,y1), (x2,y2) ):
	if x1 == x2 or y1 == y2: return True
	if abs(x1-x2) == abs(y1-y2): return True
	return False
```

## Part III: Dubins Path Planning



