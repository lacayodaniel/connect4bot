import sys
import time

INF = sys.maxsize - 1
NINF = -INF
BOT_NAME = ""
DEPTH = 2
ROWS = 6
COLS = 7
NEXTP = 1


state_count = 0  # bookkeeping to help track how efficient agents' search methods are running

def init_board(): return tuple([ tuple([0]*COLS) ]*ROWS)

def init_board_from_test(b):
    # very important to note this asumes b is [6][7]int array

    # potentially support boards of any size
    # rows = len(b)
    # cols = len(b[0])
    return tuple([ tuple(b[r]) for r in range(ROWS) ])

def next_player(board): return 1 if (sum(sum(row) for row in board) % 2) == 0 else -1 # 1 for Player 1, -1 for Player 2

def is_full(board): return moves_left(board) <= 0

def moves_left(board): return sum(sum([1 if x == 0 else 0 for x in row]) for row in board)

def _create_successor(board, col):
    global state_count
    successor_board = [ list(row) for row in board ]
    row = 0
    while (successor_board[row][col] != 0):
        row += 1
    if row >= ROWS:
        raise Exception("Illegal successor: {}, {}".format(col, board))
    successor_board[row][col] = next_player(board)
    state_count += 1
    return successor_board

def successors(board):
    move_states = [] # [(col, successor_board)]
    for col in range(COLS):
        if board[ROWS-1][col] == 0:
            move_states.append((col, _create_successor(board,col)))
    return move_states

def get_rows(board): return [[c for c in r] for r in board]

def get_cols(board): return list(zip(*board))

def get_diags(board):

    b = [None] * (len(board) - 1)
    grid_forward = [b[i:] + r + b[:i] for i, r in enumerate(get_rows(board))]
    forwards = [[c for c in r if c is not None] for r in zip(*grid_forward)]
    grid_back = [b[:i] + r + b[i:] for i, r in enumerate(get_rows(board))]
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

def scores(board):
    p1_score = 0
    p2_score = 0
    runs = get_rows(board) + get_cols(board) + get_diags(board)
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

def utility(board):
    s1, s2 = scores(board)
    return s1 - s2

def print_board(board):
    symbols = { -1: "O", 1: "X", 0: "-", -2: "#" }
    s = ""
    for r in range(ROWS-1, -1, -1):
        s += "\n"
        for c in range(COLS):
            s += "  " + symbols[board[r][c]]

    s += "\n  " + "." * (COLS * 3 - 2) + "\n"
    for c in range(COLS):
        s += "  " + str(c)
    s += "\n"
    print(s)

def get_human_move(state):
    move__state = dict(successors(state))
    prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
    move = None
    while move not in move__state:
        try:
            move = int(input(prompt))
        except ValueError:
            continue
    return move, move__state[move]

def get_move(state):
    """Select the best available move, based on minimax value."""
    moves = []
    nextp = next_player(state) # says who's move it is for the state
    best_util = NINF if nextp == 1 else INF
    best_move = None
    best_state = None

    for move, state in successors(state):
        moves.append(move)
        util = prune_minimax(state, DEPTH)
        if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
            best_util, best_move, best_state = util, move, state
    return best_move, best_state

def prune_minimax(state, depth):
    maxPlayer = True if next_player(state) == -1 else False
    return alphabeta(state, NINF, INF, maxPlayer, depth)

def alphabeta(state, alpha, beta, maxPlayer, depth):
        if ((depth == 0) or is_full(state)):
            return utility(state)

        # given, state has possible successors
        if maxPlayer:
            # currently P1's move; maximize successor states
            val = NINF
            for move, succ_state in successors(state):
                val = max(val, alphabeta(succ_state, alpha, beta, False, depth-1))
                if val >= beta:
                    break
                alpha = max(alpha,val)
            return val
        else:
            # currently P2's move; minimize successor states
            val = INF
            for move, succ_state in successors(state):
                val = min(val, alphabeta(succ_state, alpha, beta, True, depth-1))
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

def play_game(state):
    print_board(state)

    turn = 0
    p1_state_count, p2_state_count = 0, 0
    run = True
    nextp = next_player(state)

    while run:
        # player = player1 if next_player(state) == 1 else player2

        state_count_before = state_count
        
        if (nextp):
            # computer move
            a = time.time()
            move, state_next = get_move(state)
            print(time.time()-a, "seconds")
        else:
            move, state_next = get_human_move(state)

        state_count_after = state_count
        states_created = state_count_after - state_count_before

        if (nextp):
            p1_state_count += states_created
        else:
            p2_state_count += states_created

        print("Turn {}:".format(turn))
        print("Player {} generated {} states".format(1 if nextp else 2, states_created))
        print("Player {} moves to column {}".format(1 if nextp else 2, move))
        print_board(state_next)
        print("Current score is:", scores(state_next), "\n\n")

        turn += 1
        state = state_next
        if is_full(state):
            run = False
        nextp = 1 - nextp

    score1, score2 = scores(state)
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
    start_state = init_board()
    print_board(start_state)
    play_game(start_state)
