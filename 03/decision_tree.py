import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random varialbe for decision trees.
    Utility function to compute the entropy (wich is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """
    # INSERT YOUR CODE HERE.
    entropy = 0.

    num_goal_states = len(goal[1]) # In case we have many output classes, we 
                                   # must record the number of possible goal states.
    p_goal_state = [0]*num_goal_states # Variable to store probability of each goal state.

    for i in range(num_goal_states): # Iterating through each of the possible goal states
        for j in examples[:,-1]: # Iterating through each of the goaal states recorded in `examples`
            if j == i: 
                p_goal_state[i] += 1/len(examples[:,-1]) # If they match, we increment the probability of the
                                                         # given goal state

        if p_goal_state[i] != 0: # If statement to avoid log(0) error.
            entropy -= p_goal_state[i]*np.log2(p_goal_state[i]) # incrementing entropy

    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    cond_entropy = 0.0

    d = len(attribute[1]) # number of possible attribute states

    for k in range(d): # iterating through each attribute state
        # Calculate probability of attribute class k
        prob_class_k = np.sum(examples[:,col_idx] == k)/len(examples[:,col_idx])
        # Mask for rows of `examples` corresponding to attribute = k
        mask = (examples[:,col_idx] == k)
        # Calculating the in-class entropy w.r.t. goal state based on the mask. 
        class_entropy = dt_entropy(goal, examples[mask,:])
        # Incrementing total entropy arising from the condition.
        cond_entropy += prob_class_k*class_entropy

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.
    info_gain = 0.
    info_gain += dt_entropy(goal, examples) # Adding current entropy w.r.t. goal
    # Subtracting conditional entropy upon attribute split. 
    info_gain -= dt_cond_entropy(attribute, col_idx, goal, examples)

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    intrinsic_info = 0.

    d = len(attribute[1]) # Number of possible values for `attribute`.

    for k in range(d): # Iterating through each possible value for `attribute`.
        # pn represents value of `p + n`
        pn = len(examples[:,col_idx])
        # pknk represents value of `p_k + n_k`
        pknk = np.sum(examples[:,col_idx] == k)
        if pknk > 0:
            # if pknk is greater than zero, we increment intrinsic information accordingly.
            intrinsic_info -= (pknk/pn)*np.log2(pknk/pn) 

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0

    # Calculating gain from the given `attribute` test.
    gain = dt_info_gain(attribute, col_idx, goal, examples)
    # Calculating the intrinsic information of the `attribute`.
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)

    gain_ratio = 0.0

    if intrinsic_info != 0: # Conditional to avoid NaN from x/0
        gain_ratio = gain/intrinsic_info # Calculating gain ratio.

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """
    # YOUR CODE GOES HERE
    node = None
    # 1. Do any examples reach this point?
    if len(examples) == 0:
        # DONE: return plurality value for parent example
        return TreeNode(parent, None, None, True, plurality_value(goal, parent.examples))

    # 2. Or do all examples have the same class/label? If so, we're done!
    if np.sum(examples[:,-1] == examples[0,-1]) == len(examples[:,-1]):
        # DONE: We are done (whatever that means...)
        return TreeNode(parent, None, None, True, examples[0,-1])
    
    # 3. No attributes left? Choose the majority class/label.
    if len(attributes) == 0:
        return TreeNode(parent, None, None, True, plurality_value(goal, examples))

    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
    best_col_score = 0
    best_col = 0
    for i in range(len(attributes)):
        # Best score?
        if score_fun(attributes[i], i, goal, examples) > best_col_score: 
            best_col_score = score_fun(attributes[i], i, goal, examples)
            best_col = i
        # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
        # leftmost) column index!
    
    # Create a new internal node using the best attribute, something like:
    node = TreeNode(parent, attributes[best_col], examples, False, 0)

    # Now, recurse down each branch (operating on a subset of examples below).
    # You should append to node.branches in this recursion
    for i in range(len(attributes[best_col][1])):
        mask = examples[:,best_col] == i # row mask to select all examples with `attribute == i`
        col_mask = np.ones(len(examples[0,:])) # column mask to select all attributes that remain 
                                               # after the above attribute test.
        col_mask[best_col] = 0
        col_mask = col_mask == 1

        # Selecing the remaining examples using the row mask and the column mask.
        exs = examples[mask,:]
        exs = exs[:,col_mask]

        # Selecting the remaining attributes by copying and popping the attribute.
        new_attr = attributes.copy()
        new_attr.pop(best_col) 
        
        # Creating branch node (recursively)
        node.branches.append(learn_decision_tree(node, new_attr, goal, exs, score_fun))

    return node


def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root


if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])

    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
