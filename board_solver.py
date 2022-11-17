import sys

INF = sys.maxsize - 1
NINF = -INF
BOT_NAME = ""
DEPTH = 2

class GameState:
    state_count = 0  # bookkeeping to help track how efficient agents' search methods are running

    def __init__(self, *args):
        if len(args) == 2:
            r, c = args
            self.num_rows = r
            self.num_cols = c
            self.board = tuple([ tuple([0]*self.num_cols) ]*self.num_rows)
        else:
            board = args[0]
            self.num_rows = len(board)
            self.num_cols = len(board[0])
            self.board = tuple([ tuple(board[r]) for r in range(self.num_rows) ])

        # 1 for Player 1, -1 for Player 2
        self._next_p = 1 if (sum(sum(row) for row in self.board) % 2) == 0 else -1
        self._moves_left = sum(sum([1 if x == 0 else 0 for x in row]) for row in self.board)

    def next_player(self):
        
        return self._next_p

    def is_full(self):
        return self._moves_left <= 0

    def _create_successor(self, col):

        successor_board = [ list(row) for row in self.board ]
        row = 0
        while (successor_board[row][col] != 0):
            row += 1
        if row >= self.num_rows:
            raise Exception("Illegal successor: {}, {}".format(col, self.board))
        successor_board[row][col] = self._next_p
        successor = GameState(successor_board)
        GameState.state_count += 1
        return successor

    def successors(self):
        
        move_states = []
        for col in range(self.num_cols):
            if self.board[self.num_rows-1][col] == 0:
                move_states.append((col, self._create_successor(col)))
        return move_states

    def get_rows(self):
        return [[c for c in r] for r in self.board]

    def get_cols(self):
        return list(zip(*self.board))

    def get_diags(self):

        b = [None] * (len(self.board) - 1)
        grid_forward = [b[i:] + r + b[:i] for i, r in enumerate(self.get_rows())]
        forwards = [[c for c in r if c is not None] for r in zip(*grid_forward)]
        grid_back = [b[:i] + r + b[i:] for i, r in enumerate(self.get_rows())]
        backs = [[c for c in r if c is not None] for r in zip(*grid_back)]
        del forwards[0]
        del forwards[0]
        del forwards[0]
        del forwards[-1]
        del forwards[-1]
        del forwards[-1]
        del backs[0]
        del backs[0]
        del backs[0]
        del backs[-1]
        del backs[-1]
        del backs[-1]
        return forwards + backs

    def scores(self):
        p1_score = 0
        p2_score = 0
        runs = self.get_rows() + self.get_cols() + self.get_diags()
        for run in runs:
            for elt, length in streaks(run):
                if (elt == 1):
                    if (length == 2):
                        p1_score += 5
                    elif (length == 3):
                        p1_score += 25
                    elif (length >= 4):
                        p1_score += 300
                elif (elt == -1): # elt = -1
                    if (length == 2):
                        p2_score += 5
                    elif (length == 3):
                        p2_score += 25
                    elif (length >= 4):
                        p2_score += 300
                    
        return p1_score, p2_score

    def utility(self):
        s1, s2 = self.scores()
        return s1 - s2

    def __str__(self):
        symbols = { -1: "O", 1: "X", 0: "-", -2: "#" }
        s = ""
        for r in range(self.num_rows-1, -1, -1):
            s += "\n"
            for c in range(self.num_cols):
                s += "  " + symbols[self.board[r][c]]

        s += "\n  " + "." * (self.num_cols * 3 - 2) + "\n"
        for c in range(self.num_cols):
            s += "  " + str(c)
        s += "\n"
        return s

class HumanAgent:
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
        best_util = NINF if nextp == 1 else INF
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
        return self.alphabeta(state, NINF, INF, maxPlayer, depth)


    def alphabeta(self, state, alpha, beta, maxPlayer, depth):
        if ((depth == 0) or state.is_full()):
            return state.utility()

        # given, state has possible successors
        if maxPlayer:
            # currently P1's move; maximize successor states
            val = NINF
            for move, succ_state in state.successors():
                val = max(val, self.alphabeta(succ_state, alpha, beta, False, depth-1))
                if val >= beta:
                    break
                alpha = max(alpha,val)
            return val
        else:
            # currently P2's move; minimize successor states
            val = INF
            for move, succ_state in state.successors():
                val = min(val, self.alphabeta(succ_state, alpha, beta, True, depth-1))
                if val <= alpha:
                    break
                beta = min(beta,val)
            return val

def streaks(lst):
    rets = [] 
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

def play_game(player1, player2, state):
    print(state)

    turn = 0
    p1_state_count, p2_state_count = 0, 0
    run = True

    while run:
        player = player1 if state.next_player() == 1 else player2

        state_count_before = GameState.state_count
        move, state_next = player.get_move(state)
        state_count_after = GameState.state_count

        states_created = state_count_after - state_count_before
        if state.next_player() == 1:
            p1_state_count += states_created
        else:
            p2_state_count += states_created

        print("Turn {}:".format(turn))
        print("Player {} generated {} states".format(1 if state.next_player() == 1 else 2, states_created))
        print("Player {} moves to column {}".format(1 if state.next_player() == 1 else 2, move))
        print(state_next)
        print("Current score is:", state_next.scores(), "\n\n")

        turn += 1
        state = state_next
        if state.is_full():
            run = False

    score1, score2 = state.scores()
    if score1 > score2:
        print("Player 1 wins! {} - {}".format(score1, score2))
    elif score1 < score2:
        print("Player 2 wins! {} - {}".format(score1, score2))
    else:
        print("It's a tie. {} - {}".format(score1, score2))
    print("Player 1 generated {} states".format(p1_state_count))
    print("Player 2 generated {} states".format(p2_state_count))
    print("")
    return score1, score2

if __name__ == "__main__":
    players = []
    players.append(MinimaxPruneAgent())
    players.append(HumanAgent())
    start_state = GameState(6, 7)
    play_game(players[0], players[1], start_state)
