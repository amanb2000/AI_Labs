from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    =============================
    === MATHEMATICAL APPROACH ===
    =============================

    Here I implement a generic tabular q-learning approach with exponentially
    decaying exploration probability.

    The state space for my agent consists of the last `MEM` actions by `self` and
    the last `MEM` actions by the opponent.

    The agents policy `Q` maps tuples consisting of the `state` to the 
    expected value of each possible action for the next round (rock, paper, or
    scissors).

    By the definition of Q-learning, the following equation is to be satisfied:

    $$Q(s, a) = R_{t+1} + \gamma \max_{a'} Q(s',a')$$

    Where $\gamma$ is the future reward discount rate.  

    The update policy used herein will be:

    $$Q(s_t, a_t) \larr Q(s_t, a_t) + \alpha[R_{t+1} + \gamma \max_{a'}(Q(s', a') - Q(s_t, a_t))$$

    Where $\alpha$ is the learning rate. To determine the next move, the probability of
    a given action will be proportional to its Q-value. This probabilistic method will
    be taken with probability (1-exploration_probability). Otherwise, a random move 
    will be selected.

    exploration_probability will be a function of the iteration as follows:
    

    $$exploration_probability = exp{iteration/d}$$

    Where d is a tunable exponential decay constant. 

    ====================================
    === DATASTRUCTURE IMPLEMENTATION ===
    ====================================

    The policy is a function of some number of discrete variables. Every previous 
    action by the agent itself and every previous action by the opponent is either
    0, 1, or 2. This also holds true for each of the next possible actions.

    Simply concatenating all of these values forms a base-3 number that can easily
    be converted into base-10 to form an array index. This approach is advantageous
    because of the speed of array indexing and the simplicity of the policy.

    The Q-values will all be initialized to zero. 
     """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.MEM = 4
        self.q_size = 3**(self.MEM*2 + 1)

        self.Q = np.zeros(self.q_size)
        self.my_moves = np.zeros(self.MEM)
        self.others_moves = np.zeros(self.MEM)
        
        self.alpha = 0.1
        self.gamma = 1

        self.non_prob_rate = 200 # portion of the time we just take the action
                                 # with highest q-value.

        # TODO: initialize Q-values to actual expected values for 
        #       each move given a random opponent. 

        self.init_bias = 3

        rock_EV_0 = np.sum(self.game_matrix[0,:])*self.init_bias
        paper_EV_0 = np.sum(self.game_matrix[1,:])*self.init_bias
        scissors_EV_0 = np.sum(self.game_matrix[2,:])*self.init_bias

        for i in range(len(self.Q)):
            if i % 3 == 0:
                self.Q[i] = rock_EV_0
            elif i % 3 == 1:
                self.Q[i] = paper_EV_0
            else:
                self.Q[i] = scissors_EV_0

        self.iter = 0

    def get_Q(self, action, mmoves, omoves):
        """
        Function to get the Q-value of a particular `action` based on 
        self.Q, self.my_moves, and self.others_moves.

        idx = action * 3^0 + my_moves[0] * 3^1 + others_moves[0]*3^2 + ...

        :param action: Action in {0, 1, 2}.
        """

        idx = action
        xp = 1

        for i in mmoves:
            idx += 3**xp
            xp += 1

        for i in omoves:
            idx += 3**xp
            xp += 1
        
        return idx, self.Q[idx]


    def get_prob_action(self):
        prob = np.zeros(3)
        for action in (0, 1, 2):
            _, prob[action] = self.get_Q(action, self.my_moves, self.others_moves)

        prob -= np.max(prob)
        prob = np.exp(prob)/np.sum(np.exp(prob))

        return prob


    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """

        r = np.random.random()
        probs = self.get_prob_action()

        if np.random.random() > np.exp(-self.iter/self.non_prob_rate):
            return np.argmax(probs)

        # print('\tprobs: ', probs)
        if r < probs[0]:
            return 0
        elif r < probs[0] + probs[1]:
            return 1

        return 2        

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        self.iter+=1
        NEW_my_moves = np.roll(self.my_moves, 1)
        NEW_others_moves = np.roll(self.others_moves, 1)
        NEW_my_moves[0] = my_move
        NEW_others_moves[0] = other_move

        R = self.game_matrix[my_move, other_move]

        cur_Q_idx, _ = self.get_Q(my_move, self.my_moves, self.others_moves)

        next_Qs = np.zeros(3)
        for i in range(3):
            next_Q_idx, nq = self.get_Q(i, NEW_my_moves, NEW_others_moves)
            next_Qs[i] = nq

        max_next_Q = np.max(next_Qs)

        # def get_Q(self, action, mmoves, omoves):

        self.Q[cur_Q_idx] = self.Q[cur_Q_idx] + self.alpha*(R + self.gamma*max_next_Q - self.Q[cur_Q_idx])

        self.my_moves = NEW_my_moves
        self.others_moves = NEW_others_moves
         
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.my_moves = np.zeros(self.MEM)
        self.others_moves = np.zeros(self.MEM)

        rock_EV_0 = np.sum(self.game_matrix[0,:])*self.init_bias
        paper_EV_0 = np.sum(self.game_matrix[1,:])*self.init_bias
        scissors_EV_0 = np.sum(self.game_matrix[2,:])*self.init_bias

        for i in range(len(self.Q)):
            if i % 3 == 0:
                self.Q[i] = rock_EV_0
            elif i % 3 == 1:
                self.Q[i] = paper_EV_0
            else:
                self.Q[i] = scissors_EV_0

        pass


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    game_matrix = np.array([[0.0, -9.0, 1.0],
                            [9.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    print("\n\n")
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))

    student_player = StudentAgent(game_matrix)
    student_score, uniform_score = play_game(student_player, uniform_player, game_matrix)

    print("\n\n")
    print("Your player's score: {:}".format(student_score))
    print("Uniform player's score: {:}".format(uniform_score))

    print("\n\n")
    student_player = StudentAgent(game_matrix)
    student1_score, student2_score = play_game(student_player, student_player, game_matrix)
    print("Student 1 score: {}".format(student1_score))
    print("Student 2 score: {}".format(student2_score))
    print(student_player.Q)
