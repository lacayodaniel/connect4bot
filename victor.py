import copy
import numpy as np
import multiprocessing
from functools import lru_cache
import bz2
from constants import even_tree_compressed, odd_tree_compressed


class DecisionTree(object):
    """This class contains the main object used throughout this project: a decision tree. It contains methods
    to visualise and evaluate the trees."""

    def __init__(self, right=None, left=None, label='', value=None):
        """Create a node of a decision tree"""
        self.right = right
        '''right child, taken when a sample[`decisiontree.DecisionTree.label`] > `decisiontree.DecisionTree.value`'''
        self.left = left
        '''left child, taken when sample[`decisiontree.DecisionTree.label`] <= `decisiontree.DecisionTree.value`'''
        self.label = label
        '''string representation of the attribute the node splits on'''
        self.value = value
        '''the value where the node splits on (if `None`, then we're in a leaf)'''

    def evaluate(self, feature_vector):
        """Create a prediction for a sample (using its feature vector)

        **Params**
        ----------
          - `feature_vector` (pandas Series or dict) - the sample to evaluate, must be a `pandas Series` object or a
          `dict`. It is important that the attribute keys in the sample are the same as the labels occuring in the tree.

        **Returns**
        -----------
            the predicted class label
        """
        if self.value is None:
            return self.label
        else:
            # feature_vector should only contain 1 row
            if feature_vector[self.label] < self.value:
                return self.left.evaluate(feature_vector)
            else:
                return self.right.evaluate(feature_vector)

    def serialize(self):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def serializeHelper(node):
            if node.value is not None:
                vals.append(str(node.label)+'<'+str(int(np.floor(node.value) + 1)))
                serializeHelper(node.left)
                serializeHelper(node.right)
            else:
                vals.append(str(node.label))
        vals = []
        serializeHelper(self)
        return ' '.join(vals)

    @staticmethod
    def deserialize(tree_string):
        nodes = tree_string.split()
        nr_nodes = len(nodes)
        def deserializeHelper(counter):
            if counter < nr_nodes:
                node = str(nodes[counter])
                if '<' in node:
                    label, value = node.split('<')
                    dt_node = DecisionTree(label=int(label), value=int(value))

                    node, counter = deserializeHelper(counter + 1)
                    if node is not None:
                        dt_node.left = node

                    node, counter = deserializeHelper(counter)
                    if node is not None:
                        dt_node.right = node

                    return dt_node, counter
                else:
                    return DecisionTree(label=node), counter + 1
            else:
                return None, counter
        #print('RESULT:', deserializeHelper(0)[0])
        return deserializeHelper(0)[0]


def get_position_mask_bitmap(board, player):
    """Convert the game board numpy representation
    to a bitmap (encoded as integer). Store a bitmap
    for the positions of player his tokens and a
    bitmap for occupied cells. The order of
    the bits is the following (0-bit is least significant):
    .  .  .  .  .  .  .
    5 12 19 26 33 40 47
    4 11 18 25 32 39 46
    3 10 17 24 31 38 45
    2  9 16 23 30 37 44
    1  8 15 22 29 36 43
    0  7 14 21 28 35 42

    Args:
        board (numpy array)
        player (int (1 or 2))

    Return:
        position (int): token bitmap
        mask (int):  occupied cell bitmap
    """
    position, mask = '', ''
    for j in range(6, -1, -1):
        mask += '0'
        position += '0'
        for i in range(0, 6):
            mask += ['0', '1'][board[i, j] != 0]
            position += ['0', '1'][board[i, j] == player]
    return int(position, 2), int(mask, 2)


def print_bitmap(bitmap):
    """Print out the bitmap (7x7), encoded as int"""
    bitstring = bin(bitmap)[2:]
    bitstring = '0'*(49 - len(bitstring)) + bitstring
    for i in range(7):
        print(bitstring[i::7][::-1])


@lru_cache(maxsize=None)
def is_legal_move(mask, col):
    """If there is no 1-bit in the highest row
    of column `col`, then we can move there"""
    return (mask & (1 << 5 << (col*7))) == 0


@lru_cache(maxsize=None)
def make_move(position, mask, col):
    """Flip the highest zero-bit in column `col`
    of the mask bitmap and flip the bits in the
    position bitmap for the opponent to make a move"""
    new_position = position ^ mask
    new_mask = mask | (mask + (1 << (col*7)))
    return new_position, new_mask


@lru_cache(maxsize=None)
def connected_four(position):
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True

    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True

    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True

    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True

    # Nothing found
    return False


@lru_cache(maxsize=None)
def possible(mask):
    bottom_mask = 4432676798593
    board_mask = bottom_mask * 63
    return (mask + bottom_mask) & board_mask


@lru_cache(maxsize=None)
def compute_winning_position(position, mask):
    # Vertical
    r = (position << 1) & (position << 2) & (position << 3)

    # Horizontal
    p = (position << 7) & (position << 14)
    r |= p & (position << 21)
    r |= p & (position >> 7)
    p >>= 21
    r |= p & (position << 7)
    r |= p & (position >> 21)

    # Diagonal \
    p = (position <<  6) & (position << 12)
    r |= p & (position << 18)
    r |= p & (position >> 6)
    p >>= 18
    r |= p & (position << 6)
    r |= p & (position >> 18)

    # Diagonal /
    p = (position << 8) & (position << 16)
    r |= p & (position << 24)
    r |= p & (position >> 8)
    p >>= 24
    r |= p & (position << 8)
    r |= p & (position >> 24)

    return r & (4432676798593 * 63 ^ mask)# & (mask + bottom_mask)


@lru_cache(maxsize=None)
def get_columns_with_bit(bmp):
    cols = [63, 8064, 1032192, 132120576, 16911433728, 2164663517184, 277076930199552]
    return [i for i in range(len(cols)) if cols[i] & bmp]


@lru_cache(maxsize=None)
def possible_non_losing_moves(position, mask, possible_mask):
    opponent_win = compute_winning_position(position ^ mask, mask)
    forced_moves = possible_mask & opponent_win
    if forced_moves:
        if forced_moves & (forced_moves - 1):
            return []
        else:
            possible_mask = forced_moves
    return get_columns_with_bit(possible_mask & ~(opponent_win >> 1))


@lru_cache(maxsize=None)
def popcount(m):
    c = 0
    while m:
        m &= (m - 1)
        c += 1
    return c


@lru_cache(maxsize=2048)
def insertion(M):
    idx = list(range(len(M)))
    M = list(M)
    for i in range(1,len(M)):
        if M[i] >= M[i-1]:
            continue
        for j in range(i):
            if M[i] < M[j]:
                M[j],M[j+1:i+1] = M[i],M[j:i]
                idx[j],idx[j+1:i+1] = idx[i],idx[j:i]
                break
    return M, idx


def alphabeta_search(position, mask, ub_cache=None, lb_cache=None, valid_actions=[]):
    """Search game to determine best action; use alpha-beta pruning.
    this version searches all the way to the leaves."""

    # First explore columns in the center
    actions = [3, 2, 4, 1, 5, 0, 6]

    # Initialize our cache (dynamic programming)
    if ub_cache is None:
        ub_cache = {}
    if lb_cache is None:
        lb_cache = {}

    # Functions used by alphabeta
    def max_value(position, mask, alpha, beta, moves):
        possible_mask = possible(mask)

        # Check if game terminates
        if connected_four(position):
            return int(-np.ceil((43 - moves) / 2))
        elif compute_winning_position(position, mask) & possible_mask:
            return int(np.ceil((42 - moves) / 2))

        key = position + mask

        # Try to get an upper bound for this position from cache
        try:
            v = ub_cache[key]
            if beta >= v:
                beta = v
        except KeyError as e:
            pass
        
        v = max(alpha, (-42 + moves)//2)
        if v >= alpha:
            alpha = v

        if alpha >= beta:
            return beta

        
        if moves < 30:
            # This helps for positions close to the end
            _actions = possible_non_losing_moves(position, mask, possible_mask)
            if not len(_actions): 
                return -int(np.ceil((41 - moves) / 2))

            nr_winning_actions_per_col = [0]*7
            for col in actions:
                new_position, new_mask = make_move(position, mask, col)
                nr_winning_actions_per_col[col] = popcount(compute_winning_position(new_position ^ new_mask, 
                                                                                    new_mask))

            # This can possibly be replaced with np.argsort
            _actions = insertion(tuple(nr_winning_actions_per_col))[1][::-1]
        else:
            _actions = actions

        for col in _actions:
            if is_legal_move(mask, col):
                new_position, new_mask  = make_move(position, mask, col)
                v = max(v, min_value(
                             new_position, new_mask,
                             alpha, beta, moves + 1
                           )
                        )
                if v >= beta:
                    return v
                if v > 0:  # We found a move towards a win (not the optimal one)
                    return v
                alpha = max(alpha, v)

        # Put our new upper bound in the cache
        ub_cache[key] =  alpha
        return v

    def min_value(position, mask, alpha, beta, moves):
        possible_mask = possible(mask)

        # Check if game terminates
        if connected_four(position):
            return int(np.ceil((43 - moves) / 2))
        elif compute_winning_position(position, mask) & possible_mask:
            return int(-np.ceil((42 - moves) / 2))

        key = position + mask

        try:
            v = lb_cache[key]
            if alpha <= v:
                alpha = v
        except KeyError as e:
            pass

        v = min(beta, (42 - moves)//2)
        if v <= beta:
            beta = v

        if alpha >= beta:
            return alpha

        if moves < 30:
            # This helps for positions close to the end
            _actions = possible_non_losing_moves(position, mask, possible_mask)
            if not len(_actions): 
                return int(np.ceil((41 - moves) / 2))

            nr_winning_actions_per_col = [0]*7
            for col in actions:
                new_position, new_mask = make_move(position, mask, col)
                nr_winning_actions_per_col[col] = popcount(compute_winning_position(new_position ^ new_mask, new_mask))

            # This can possibly be replaced with np.argsort
            _actions = insertion(tuple(nr_winning_actions_per_col))[1][::-1]
        else:
            _actions = actions

        for col in _actions:
            if is_legal_move(mask, col):
                new_position, new_mask = make_move(position, mask, col)
                v = min(v, max_value(
                             new_position, new_mask,
                             alpha, beta, moves + 1
                           )
                        )
                if v <= alpha:
                    return v
                if v < 0:  # We found a move towards a win (not the optimal one)
                    return v
                beta = min(beta, v)

        lb_cache[key] =  beta
        return v


    # Body of alphabeta_cutoff_search:
    n_bits = popcount(mask)
    best_score = -(42 - n_bits)//2
    beta = (42 - n_bits)//2
    best_action = None

    nr_winning_actions_per_col = [0]*7
    for col in actions:
        if col in valid_actions and is_legal_move(mask, col):
            new_position, new_mask = make_move(position, mask, col)
            if connected_four(new_position ^ new_mask):
                return  col, int(np.ceil((43 - n_bits - 1) / 2))
            nr_winning_actions_per_col[col] = popcount(compute_winning_position(new_position ^ new_mask, 
                                                                                new_mask))

    actions = insertion(tuple(nr_winning_actions_per_col))[1][::-1]

    for col in actions:
        if is_legal_move(mask, col):
            new_position, new_mask  = make_move(position, mask, col)

            v = min_value(new_position, new_mask, best_score, beta, n_bits + 1)
            if v > best_score:
                best_score = v
                best_action = col

    return best_action, best_score


def opponent_can_finish(board, action, player, turn):
    b = copy.deepcopy(board)
    b[(b[:, action] == 0).sum() - 1, action] = player
    for a in np.flatnonzero(b[0, :] == 0):
        if turn == 1 and length_pot_move(b, a, 3 - player) >= 4:
            return True
        elif turn > 1 and player_can_finish(b, a, 3 - player, turn - 1):
            return True
    return False


def player_can_finish(board, action, player, turn):
    b = copy.deepcopy(board)
    b[(b[:, action] == 0).sum() - 1, action] = player
    for a in np.flatnonzero(b[0, :] == 0):
        if length_pot_move(b, a, 3 - player) >= 4:
            return False
        if not opponent_can_finish(b, a, 3 - player, turn):
            return False
    return True


def length_pot_move(board, action, player):
    b = copy.deepcopy(board)
    pos = ((b[:, action] == 0).sum() - 1, action)
    b[pos] = player
    return max([length(b, pos, d) for d in [(1, 0), (0, 1), (1, 1), (1, -1)]])


def length(board, pos, d):
    player = board[pos]
    l = 1
    pos_t = pos
    for i in range(3):
        pos_t = (pos_t[0] + d[0], pos_t[1] + d[1])
        if 0 <= pos_t[0] <= 5 and 0 <= pos_t[1] <= 6 and board[pos_t] == player:
            l += 1
        else:
            break
    pos_t = pos
    for i in range(3):
        pos_t = (pos_t[0] - d[0], pos_t[1] - d[1])
        if 0 <= pos_t[0] <= 5 and 0 <= pos_t[1] <= 6 and board[pos_t] == player:
            l += 1
        else:
            break
    return l


def center_action(actions):
    for action in [3, 2, 4, 1, 5, 0, 6]:
        if action in actions:
            return action


def generate_fast_move(board, player):
    # Determine the set of valid actions
    actions = np.flatnonzero(board[0, :] == 0)

    # Can I finish it in this turn?
    finishers = []
    for action in actions:
        if length_pot_move(board, action, player) >= 4:
            finishers.append(action)
    if finishers:
        action = center_action(finishers)
        return [action]

    # Can the opponent finish it in the next turn?
    actions = [action for action in actions if not opponent_can_finish(board, action, player, 1)]
    if len(actions) == 1:
        return actions

    # Can I finish it in the next turn?
    finishers = []
    for action in actions:
        if player_can_finish(board, action, player, 1):
            finishers.append(action)
    if finishers:
        action = center_action(finishers)
        return [action]
    
    # Can the opponent finish it in the second to next turn?
    actions = [action for action in actions if not opponent_can_finish(board, action, player, 2)]
    if len(actions) == 1:
        return actions

    # Return all valid actions
    return actions


def generate_slow_move(board, player, saved_state, q, valid_actions=[]):
    """Contains all code required to generate a move,
    given a current game state (board & player)

    Args:

        board (2D np.array):    game board (element is 0, 1 or 2)
        player (int):           your player number (token to place: 1 or 2)
        saved_state (object):   returned value from previous call

    Returns:

        action (int):                   number in [0, 6]
        saved_state (optional, object): will be returned to you the
                                        next time your function is called

    """
      
    position, mask = get_position_mask_bitmap(board, player)
    q.put(alphabeta_search(position, mask, valid_actions=valid_actions))


def mirror(s):
    return int(''.join([str(8 - int(i)) for i in str(s)]))


def generate_move(board, player, state):
    b = board[:, :]
    if state is None:
        if np.sum(b != 0) == 0:

            tree_string = str(bz2.decompress(odd_tree_compressed))[2:-1]
            dt = DecisionTree.deserialize(tree_string)
            opt = 3
            b[(b[:, opt] == 0).sum() - 1, opt] = player
            state = ("", dt, b)
            #print('Fast (%i)'%opt)
            return opt, state
        else:
            
            tree_string = str(bz2.decompress(even_tree_compressed))[2:-1]
            dt = DecisionTree.deserialize(tree_string)
            a = np.argmax(b[5,:])
            opt = [3, 2, 3, 3, 3, 4, 3][a]
            b[(b[:, opt] == 0).sum() - 1, opt] = player
            state = (str(a), dt, b)
            #print('Fast (%i)'%opt)
            return opt, state
    
    s = state[0] + str(np.where(state[2] - b != 0)[0][0])
    dt = state[1]
    
    actions = generate_fast_move(b, player)
    if len(actions) == 1:
        opt = actions[0]
        #print('Fast (%i)'%opt)
        b[(b[:, opt] == 0).sum() - 1, opt] = player
        state = (s, dt, b)
        return opt, state
    
    try:
        q = multiprocessing.Queue()
        proc = multiprocessing.Process(target=generate_slow_move, args=[b, player, '', q, actions])
        proc.start()
        proc.join(timeout=0.7)
        proc.terminate()
        if q.qsize():
            opt = q.get()[0]
            if opt > -1:
                #print("Slow (%i)" % opt)
                b[(b[:, opt] == 0).sum() - 1, opt] = player
                state = (s, dt, b)
                return opt, state
        raise
    except Exception as e:
        # Timeout occurred, so search for the state
        if mirror(int(s)) < int(s):
            # Convert sequence to list and pad 0's
            #print(str(mirror(int(s))))
            feature_vector =  list(map(int, list(str(mirror(s))))) + [0]*(15-len(str(mirror(int(s)))))
            opt = 7 - int(dt.evaluate(feature_vector))
        else:
            # Convert sequence to list and pad 0's
            feature_vector =  list(map(int, list(s))) + [0]*(15-len(s))
            opt = int(dt.evaluate(feature_vector)) - 1  
                

        if opt in actions:
            b[(b[:, opt] == 0).sum() - 1, opt] = player
            state = (s, dt, b)
            #print('Tree (%i)'%opt)
            return opt, state
        
        # Find column that results in the most open-ended connect-four's
        position, mask = get_position_mask_bitmap(b, player)
        best_score = float('-inf')
        opt = 0
        for col in actions:
            new_position, new_mask = make_move(position, mask, col)
            score = popcount(compute_winning_position(new_position ^ new_mask, new_mask))
            if score > best_score:
                best_score = score
                opt = col
                
        b[(b[:, opt] == 0).sum() - 1, opt] = player
        state = (s, dt, b)
        #print('Heur (%i)'%opt)
        return opt, state

def my_move(board, action, player):
    pos = ((board[:, action] == 0).sum() - 1, action)
    board[pos] = player
    return max([length(board, pos, d) for d in [(1, 0), (0, 1), (1, 1), (1, -1)]]) >= 4


def generate_board(s):
    board = np.zeros([6, 7], dtype=int)
    for i in range(len(str(s))):
        if i == 0:
            action_1 = 3
        else:
            m_1 = int(str(s)[:i])
            m_2 = int(mirror(str(s)[:i]))
            if m_1 in d:
                action_1 = d[m_1] - 1
            elif m_2 in d:
                action_1 = d[m_2] - 1
            else:
                print("ERROR: %i nor %i available in dict" % (m_1, m_2))
        action_2 = int(str(s)[i]) - 1
        pos = ((board[:, action_1] == 0).sum() - 1, action_1)
        board[pos] = 1
        pos = ((board[:, action_2] == 0).sum() - 1, action_2)
        board[pos] = 2
    return board

import time
def main():
    board = np.zeros([6, 7], dtype=int)
    s1, s2 = None, None
    for i in range(21): # 21 turns to win with perfect play
        # computer goes first
        t = time.time()
        a, s1 = generate_move(copy.deepcopy(board), 1, s1)
        f = my_move(board, a, 1)
        print(time.time() - t)
        print(board)
        if f:
            print("Game won by player 1")
            break
        t = time.time()
        a = int(input('Give a move:'))
        f = my_move(board, a, 2)
        print(board)
        if f:
            print("Game won by player 2")
            break

if __name__ == '__main__':
    main()