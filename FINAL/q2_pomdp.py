# Any additional import statements will be removed by Autolab
import numpy as np


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
        pass

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

        return np.random.randint(3)  # Replace this random policy with your solution

    def compute_action(self, percept: int) -> int:
        """
        Compute the next action given a percept.

        :param percept: noisy measurement of the state provided by the POMDP
        :return: integer corresponding to an action (only 0, 1, and 2 are valid)
        """

        ### STUDENT CODE GOES HERE
        # Update your belief from the percept (and last action...), then select and return the next action to take
        # TIP: you will probably want to store the last action taken as well...

        return np.random.randint(3)  # Replace this random policy with your solution


if __name__ == '__main__':
    # This is how Autolab will test your solver
    n_runs = 10000  # Number of runs - your score will be based on your average return over these runs

    pomdp = get_chatbot_pomdp()  # Instantiate the chatbot POMDP
    solver = ChatbotSolver(pomdp.T, pomdp.P)  # Instantiate your solver

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

    # Your grade will be based on a linear interpolation between the score of two reference solvers
