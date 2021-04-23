from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code

    db = set(known_symbols) # instantiating our database with our known symbols

    added = True # boolean for if our knowledge base is still changing or if we have hit the limit.

    while added: # while we are yet to converge,
        added = False
        processed = [] # array for storing processed clause indexes in the KB
        for i, clause in enumerate(KB_clauses): # Iterating through all remaining unprocessed clauses.
            if db.issuperset(clause.body) and clause.conclusion not in db: # If the body of the clause is in the KB and the conclusion is NOT, it's new!
                db.add(clause.conclusion) # Adding the conclusion to our knowledge base.
                added = True # Updating our variable for documenting convergence (lack thereof).
                processed.append(i-len(processed)) # Adding the ID of the processed clause (minus i because of offset introduced by pop() later)

        if query in db: # Checking if we reached our goal
            return True

        for p in processed: # Popping the clauses from the KB that we have processed.
            KB_clauses.pop(p)

    return False # Returning false if we exhausted our resources and were not able to prove the query.
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
