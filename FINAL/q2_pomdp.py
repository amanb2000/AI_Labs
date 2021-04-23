# Any additional import statements will be removed by Autolab
import numpy as np
# from tqdm import tqdm


def categorical_sample_index(p: np.ndarray) -> int:
    """
    Sample a categorical distribution.

    :param p: a categorical distribution's probability mass function (i.e., p[idx] is the probability of this function
              returning idx for an integer 0 <= idx < len(p)). I.e., np.sum(p) == 1 and p[idx] >= 0 for 0<=idx<len(p).
    :return: index of a sample weighted by the categorical distribution described by p
    """
    P = np.cumsum(p)
    sample = np.random.rand()
    return np.argmax(P > sample)


class POMDP:
    """
    Partially observable Markov decision process
    """
    def __init__(self, S, S_terminal, A, T, R, P):
        self.S = S
        self.S_terminal = S_terminal
        self.S_non_terminal = {key: val for key, val in zip(self.S.keys(), self.S.values())
                               if key not in self.S_terminal}
        self.A = A
        self.T = T
        self.R = R
        self.P = P

        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.percept_history = []

    def set_initial_state(self, state):
        assert state in self.S, "ERROR: initial state {:} is not valid.".format(state)
        self.state_history = [state]
        self.action_history = []
        self.reward_history = []
        self.percept_history = []

    def take_action(self, action) -> (float, int):
        assert action in self.A, f"ERROR: action '{action}' not permitted."
        s_current = self.state_history[-1]
        self.action_history.append(action)
        # Sample the next state
        T_action = self.T[s_current, action, :]
        s_next = categorical_sample_index(T_action)
        self.state_history.append(s_next)
        reward = self.R[s_next]
        self.reward_history.append(reward)
        percept = categorical_sample_index(self.P[s_next, :])
        self.percept_history.append(percept)

        return reward, percept


def get_chatbot_pomdp():
    # Sales chatbot POMDP's states
    S = {0: 'Hangup', 1: 'Annoyed', 2: 'Neutral', 3: 'Engaged', 4: 'Sale'}
    S_terminal = set((0, 4))

    # Sales chatbot POMDP's actions
    A = {0: 'Aggressive', 1: 'Informative', 2: 'Apologetic'}

    # Construct 3D transition tensor indexed by: State at Time t -> Action -> State at Time t+1
    T_hangup = np.zeros((len(A), len(S)))
    T_hangup[:, 0] = 1
    T_annoyed = np.array([[0.8, 0.2, 0., 0., 0.],  # When annoyed, aggression is bad
                          [0.1, 0.1, 0.8, 0., 0.],
                          [0.0, 0.0, 0.5, 0.5, 0.]])  # Apologizing is great when annoyed

    T_neutral = np.array([[0.1, 0.2, 0.3, 0.4, 0.],  # Aggression is OK when neutral
                          [0.0, 0.1, 0.2, 0.7, 0.],
                          [0.2, 0.5, 0.3, 0., 0.]])

    T_engaged = np.array([[0.0, 0.1, 0.1, 0.1, 0.7],  # When engaged, aggression works well
                          [0.0, 0.4, 0.3, 0.3, 0.0],
                          [0.2, 0.5, 0.3, 0.0, 0.0]])
    T_sale = np.zeros((len(A), len(S)))
    T_sale[:, 4] = 1
    T = np.array((T_hangup, T_annoyed, T_neutral, T_engaged, T_sale))

    # Rewards for arriving in each state
    time_reward = -0.01  # Opportunity cost of keeping the customer in the chat
    hangup_reward = -1.  # Cost of lost business from a Hangup
    sale_reward = 2.     # Value of a Sale
    R = {0: hangup_reward, 1: time_reward, 2: time_reward, 3: time_reward, 4: sale_reward}

    # Observations are exact when in a terminal state, identically noisy otherwise
    p_correct = 0.6  # Probability of observing the correct non-terminal state
    p_other = (1 - p_correct) / 2  # Probability of observing each other non-terminal state
    # Put this in matrix form
    P = p_correct * np.eye(len(S))
    for state in S_terminal:
        P[state, state] = 1.
    for state_idx in S:
        if state_idx not in S_terminal:
            for state_jdx in S:
                if state_jdx not in S_terminal:
                    if state_jdx != state_idx:
                        P[state_idx, state_jdx] = p_other

    return POMDP(S, S_terminal, A, T, R, P)  # Return a POMDP with the specified parameters


class ChatbotSolver:

    def __init__(self, T: np.ndarray, P: np.ndarray):
        # Store the transition and perception matrices
        self.T = T
        self.P = P

        print("T: ")
        print(T)

        print("P: ")
        print(P)

        ### STUDENT CODE GOES HERE - load any information that you need to implement your policy
        # You will most likely want to load a numpy array that stores the policy you learned with value iteration
        # Example: how to save an array as a .npy file (do this on your machine):
        #   lookup_table = SOME_ARRAY_YOU_COMPUTE()
        #   with open('lookup_table.npy', 'wb') as f:
        #       np.save(f, lookup_table)
        #
        # Then, include lookup_table.npy in your submission:
        #   tar cvf handin.tar q2_pomdp.py lookup_table.npy OTHER_CODE.py
        #
        # where OTHER_CODE.py includes all other code you used for Question 2 (including the code that solved question
        # 2b and generated lookup_table.npy).
        #
        # Finally, load your code in this __init__() method (with this EXACT relative path):
        # with open("../student_submission/lookup_table.npy", 'rb') as f:
        #     self.lookup_table = np.load(f).T

        # path = "U10.npy"
        path = "../student_submission/U10.npy"
        U = None
        with open(path, 'rb') as f:
            U = np.load(f, allow_pickle=True)
        lu = len(U)
        ls = 5
        
        A = np.zeros([lu, ls])

        for i in range(len(U)):
            A[i,:] = U[i][1]

        self.A = A
        self.U = U

        self.b = None
        self.last_action = None

    def restart(self, b_init: np.ndarray) -> int:
        """
        Called as the first move for each episode (interaction with the POMDP) starts.

        :param b_init: 5-vector representing the initial belief state: e.g., b_init[1] is the probability that the
                       initial state is 'Annoyed', b_init[2] is the probability that the initial state is 'Neutral'.
                       b[0] = b[4] = 0 (i.e., we never start in a terminal state).
        :return: integer in (0, 1, 2) corresponding to the first action to take
        """

        ### STUDENT CODE GOES HERE
        # You'll probably want to store the initial belief state:
        # self.b = b_init
        # Then, choose and return the first action from (0, 1, 2)
        # TIP: you will probably want to store the actions you take, or at least the previous one...

        self.b = b_init

        # Select an action
        action_idx = np.argmax(self.A @ self.b)
        action = self.U[action_idx][0]
        self.last_action = action

        return action  # Replace this random policy with your solution

    def compute_action(self, percept: int) -> int:
        """
        Compute the next action given a percept.

        :param percept: noisy measurement of the state provided by the POMDP
        :return: integer corresponding to an action (only 0, 1, and 2 are valid)
        """

        ### STUDENT CODE GOES HERE
        # Update your belief from the percept (and last action...), then select and return the next action to take
        # TIP: you will probably want to store the last action taken as well...

        # Update our belief:
        new_belief = np.zeros(self.b.shape)
        for s_p in range(5):
            new_belief[s_p] = self.P[percept, s_p] # new_belief[s_p] = P(e | s_p)
            sum_term = 0
            for s in range(5):
                sum_term += self.T[s, self.last_action, s_p] * self.b[s]# sum_term += P(s_p | s, a) b[s]
            new_belief[s_p] += sum_term

        new_belief /= np.sum(new_belief)

        self.b = new_belief
        action_idx = np.argmax(self.A @ self.b)
        action = self.U[action_idx][0]

        self.last_action = action

        return action  # Replace this random policy with your solution



# PLOTTING CODE #
def kill_dominated_truncated(U,pomdp,comb=0.01):
    # Step I: Create the big A matrix. 
    lu = len(U)
    ls = len(pomdp.S)
    A = np.zeros([lu, ls])

    for i in range(len(U)):
        A[i,:] = U[i][1]

    A = A[:,1:4]
    
    # Step II: Test where the argmax occurs for different values of P's
    argmaxes = []

    ps = np.arange(0,1,comb)

    cur_vec = np.zeros(3)

    for p0 in ps:
        for p1 in ps:
            if p0+p1 > 1:
                continue
            p2 = 1-p0-p1
            cur_vec[0] = p0
            cur_vec[1] = p1
            cur_vec[2] = p2

            argmax = np.argmax(A @ cur_vec)
            # print("argmax: ",argmax)
            if argmax not in argmaxes:
                argmaxes.append(argmax)

    print("From argmax: Length of argmax is: ",len(argmaxes))

    U_new = []

    for i in range(len(U)):
        if i in argmaxes:
            U_new.append(U[i])
    
    return U_new

def kill_dominated(U, pomdp, comb=0.05):
    # Step I: Create the big A matrix. 
    lu = len(U)
    ls = len(pomdp.S)
    A = np.zeros([lu, ls])

    for i in range(len(U)):
        A[i,:] = U[i][1]
    
    # Step II: Test where the argmax occurs for different values of P's
    argmaxes = []

    ps = np.arange(0,1,comb)

    cur_vec = np.zeros(5)

    for p0 in ps:
        for p1 in ps:
            for p2 in ps:
                for p3 in ps:
                    if p0+p1+p2+p3 > 1:
                        continue
                    p4 = 1-p0-p1-p2-p3
                    cur_vec[0] = p0
                    cur_vec[1] = p1
                    cur_vec[2] = p2
                    cur_vec[3] = p3
                    cur_vec[4] = p4

                    argmax = np.argmax(A @ cur_vec)
                    # print("argmax: ",argmax)
                    if argmax not in argmaxes:
                        argmaxes.append(argmax)

    print("From argmax: Length of argmax is: ",len(argmaxes))

    U_new = []

    for i in range(len(U)):
        if i in argmaxes:
            U_new.append(U[i])

    return U_new

def value_iterate(pomdp, level=10, comb=0.05):
    U = []
    U_p = []

    # Step 0: get a few constants.
    slen = len(pomdp.S)
    alen = len(pomdp.A)
    print("|A|: {}".format(alen))
    print("|S|: {}".format(slen))
    print("P: {}".format(pomdp.P))

    # Step I: Generating all 1-step plans in U.
    for a in pomdp.A:
        add_ary = [a, None, []] # add_ary[1] will be \vec alpha_p
        alpha_vec = np.zeros(slen)
        for s in range(slen):
            alpha_vec[s] = pomdp.R[s]
            for s_p in range(slen):
                alpha_vec[s] += pomdp.T[s,a,s_p] * pomdp.R[s_p]
        
        add_ary[1] = alpha_vec
        U.append(add_ary)
    
    print(U)

    print("Saving value of U for iteration {}...".format(1))
    with open('cache/U{}.npy'.format(1), 'wb') as f:
        np.save(f, U)
    print("Done saving.\n")

    for cnt in range(level): # We have already done the first value iteration, so we have `level-1` left.
        U_p = []
        for a in pomdp.A:
            for p1 in U:
                for p2 in U:
                    for p3 in U:
                        add_ary = [a, None, [p1, p2, p3]]
                        alpha = np.zeros(slen)
                        for s in range(slen):
                            alpha[s] = pomdp.R[s]
                            add_term = 0
                            for s_p in range(slen):
                                add_term_p1 = pomdp.T[s,a,s_p]
                                add_term_p2 = 0
                                for e in range(3): # percepts are in range 3.
                                    add_term_p2 += pomdp.P[e+1,s_p]*add_ary[2][e][1][s_p]

                                add_term += add_term_p1*add_term_p2
                            alpha[s] += add_term
                        add_ary[1] = alpha.copy()

                        U_p.append(add_ary)
        U = U_p.copy()
        print("Iteration {}: Size of U is {}".format(cnt+2, len(U_p)))
        # Cull the weak here.
        # U = kill_dominated(U, pomdp, comb=comb)
        U = kill_dominated_truncated(U, pomdp, comb=comb)
        print("After killing the weak, U is {}".format(len(U)))

        print("Saving value of U for iteration {}...".format(cnt+2))
        with open('cache/U{}.npy'.format(cnt+2), 'wb') as f:
            np.save(f, U)
        print("Done saving.\n")


if __name__ == '__main__':
    # This is how Autolab will test your solver
    generate_arrays = False

    n_runs = 10000  # Number of runs - your score will be based on your average return over these runs

    pomdp = get_chatbot_pomdp()  # Instantiate the chatbot POMDP
    solver = ChatbotSolver(pomdp.T, pomdp.P)  # Instantiate your solver

    if generate_arrays:
        print("Starting value iteration!\n")
        value_iterate(pomdp, comb=0.01, level=10)
    else: # Leaving this branch as is for testing.
        total_returns = np.zeros(n_runs)  # Record the total return of each run

        b_init = np.array([0., 0.25, 0.25, 0.5, 0.])  # The initial belief state (this will vary per run on Autolab)
        for idx in range(n_runs):
            init_state = categorical_sample_index(b_init)  # Each run, sample an initial state from the initial belief
            pomdp.set_initial_state(init_state)  # Reset the POMDP
            a0 = solver.restart(b_init)  # Restart your solver and get its first action
            total_return, percept = pomdp.take_action(a0)  # Take the first action, observe the first reward and percept
            while pomdp.state_history[-1] not in pomdp.S_terminal:  # Iterate until arriving in a terminal state
                a = solver.compute_action(percept)  # Get the next action based on the percept
                reward, percept = pomdp.take_action(a)  # Execute this action, observe the next reward and percept
                total_return += reward  # Add the reward to the total return for this run

            total_returns[idx] = total_return  # Store the total return
        print(sum(total_returns))

        # Your grade will be based on a linear interpolation between the score of two reference solvers
