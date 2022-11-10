from mimetypes import init
import random
import math


BOT_NAME = ""
DEPTH = 4


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""
    def __init__(self):
        self.moves = []
    
    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player() # says who's move it is for the state
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            self.moves.append(move)
            util = self.minimax(state, DEPTH)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state


    def minimax(self, state, depth):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        if ((depth == 0) or state.is_full()):
            return state.utility()

        # state has possible successors
        if state.next_player() == 1:
            # current P2 move so minimize successor states
            return min([self.minimax(succ_state, depth-1) for move, succ_state in state.successors()])

        # currently P1 move so maximize successor states
        return max([self.minimax(succ_state, depth-1) for move, succ_state in state.successors()])


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
    """

    def __init__(self, depth_limit):
        super().__init__()
        self.depth_limit = depth_limit

    def minimax(self, state, depth):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game
        tree that gets explored before estimating the state utilities using the evaluation()
        function.  If depth is 0, no traversal is performed, and minimax returns the results of
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.minimax_depth(state,0)


    def minimax_depth(self, state, depth):
        """This is just a helper method for minimax()."""
        if depth >= self.depth_limit:
            return self.evaluation(state)

        if state.is_full():
            return state.utility()

        # given, state has possible successors
        if state.next_player() == 1:
            # currently P2's move; minimize successor states
            return min([self.minimax_depth(succ_state,depth+1) for move, succ_state in state.successors()])

        # currently P1's move; maximize successor states
        return max([self.minimax_depth(succ_state,depth+1) for move, succ_state in state.successors()])


    """Returns: a heuristic estimate of the utility value of the state"""
    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!

        Args:
            state: a connect383.GameState object representing the current board
        """

        """determine who's turn it is at the origin
        the oppents value will be filled in the board anywhere there is a
        spot available"""
        if self.depth_limit % 2 == 0:
            player_v = state.next_player()
        else:
            player_v = -1 if state.next_player() == 1 else 1
            
        p1_score = 0
        p2_score = 0

        for run in state.get_rows()+state.get_cols()+state.get_diags():
            Hrun = list(map(lambda x: x if x != 0 else player_v, run)) # the heuristic run
            for player, length in self.streak(Hrun):
                if (player == 1) and (length >= 3):
                    p1_score += length**2
                elif (player == -1) and (length >= 3):
                    p2_score += length**2
        # print(self.print_end_state(state, player_v), p1_score-p2_score, "Move", self.moves[-1])
        return p1_score-p2_score
    
    def print_end_state(self, state, fill):
        if fill == -1:
            symbols = { -1: "O", 1: "X", 0: "O", -2: "#" }
        else:
            symbols = {-1: "O", 1: "X", 0: "X", -2: "#"}
        s = ""
        for r in range(state.num_rows-1, -1, -1):
            s += "\n"
            for c in range(state.num_cols):
                s += "  " + symbols[state.board[r][c]]

        s += "\n  " + "." * (state.num_cols * 3 - 2) + "\n"
        for c in range(state.num_cols):
            s += "  " + str(c)
        s += "\n"
        return s

    def streak(self,lst):
        """Get the lengths of all the streaks of the same element in a sequence."""
        rets = []  # list of (element, length) tuples
        prev = lst[0]
        curr_len = 1
        for curr in lst[1:]:
            if curr == prev:
                curr_len += 1
            else:
                rets.append((prev, curr_len))
                prev = curr
                curr_len = 1
        rets.append((prev, curr_len))
        return rets


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move.
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want?
    """

    def minimax(self, state, depth):
        """Determine the minimax utility value of the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the
        algorithm should do less work.  You can check this by inspecting the value of the class
        variable GameState.state_count, which keeps track of how many GameState objects have been
        created over time.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        maxPlayer = True if state.next_player() == -1 else False
        return self.alphabeta(state, -math.inf, math.inf, maxPlayer, depth)


    def alphabeta(self, state, alpha, beta, maxPlayer, depth):
        if ((depth == 0) or state.is_full()):
            return state.utility()

        # given, state has possible successors
        if maxPlayer:
            # currently P1's move; maximize successor states
            val = -math.inf
            for move, succ_state in state.successors():
                val = max(val, self.alphabeta(succ_state, alpha, beta, False, depth-1))
                if val >= beta:
                    break
                alpha = max(alpha,val)
            return val
        else:
            # currently P2's move; minimize successor states
            val = math.inf
            for move, succ_state in state.successors():
                val = min(val, self.alphabeta(succ_state, alpha, beta, True, depth-1))
                if val <= alpha:
                    break
                beta = min(beta,val)
            return val
