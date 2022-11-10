import numpy as np



def get_position_mask_bitmap(board, player):
    position, mask = '', ''
    # Start with right-most column
    for j in range(6, -1, -1):
        # Add 0-bits to sentinel 
        mask += '0'
        position += '0'
        # Start with bottom row
        for i in range(0, 6):
            mask += ['0', '1'][board[i, j] != 0]
            position += ['0', '1'][board[i, j] == player]
    return int(position, 2), int(mask, 2)

def make_move(position, mask, col):
    new_position = position ^ mask
    new_mask = mask | (mask + (1 << (col*7)))
    return new_position, new_mask

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


if __name__ == "__main__":
    height = [0, 7, 14, 21, 28, 35, 42]
    board = np.zeros(shape=(6, 7))
    position, mask = get_position_mask_bitmap(board,1)
    new_pos, new_mask = make_move(position, mask, 1)
    new_pos, new_mask = make_move(new_pos, new_mask, 0)
    new_pos, new_mask = make_move(new_pos, new_mask, 2)
    new_pos, new_mask = make_move(new_pos, new_mask, 3)
    print(connected_four(new_pos))
    print(bin(new_mask))
