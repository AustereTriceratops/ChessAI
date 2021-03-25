import os
import numpy as np
import chess

from constants import *



# parses data from the .txt file of the games into two equal-sized arrays of game strings and results
# dataset from: https://www.kaggle.com/milesh1/35-million-chess-games
def parse_raw_data(start, end):
    games = []  # contains strings of games
    results = []

    rootpath = os.getcwd()
    datapath = os.path.join(os.path.dirname(rootpath), "kaggle", "chessgames", "all_with_filtered_anotations_since1998.txt")
    # remove hardcoded path


    with open(datapath) as f:
        for i in range(end):

            line = f.readline()

            if i < start:
                continue

            if line[0] == '#':
                continue

            line = line.split('###')

            info = line[0].split(" ")
            moves = line[1]

            # grab data 
            result = info[2]
            white_elo = info[3]
            black_elo = info[4]
            fen = info[12]
            #elo = 0

            # skip "empty" games that sometimes show up
            if len(moves) <= 10:
                continue

            # skip 960 games
            if fen == 'fen_true':
                continue

            # skip game if no elo is known (is this necessary?)
            if black_elo == "None" and white_elo == "None":
                continue

            # add moves
            games.append(moves)

            # parse result string
            s = 0.5

            if result == '1-0':
                s = 1
            elif result == "0-1":
                s = 0

            results.append(s)

    return games, results


def encode_square(sq):   
    # encode into tuple from uci format (e.g. 'e4', 'h8', 'c1')
    x = LETTERS[sq[0]]
    y = int(sq[1]) -1

    return (x, y)


def encode_bit(bit):
    # encode number from 0-63 into square tuple (x, y)
    x = bit % 8
    y = int((bit - x)/8)

    return (x, y)


def encode_move(move_uci):
    # encodes a move from uci format
    # 'c3e4' -> (2, 1)
    # 'dxc8n' -> (-1, 1, 'n')
    # does not check for legality of move
    start = move_uci[:2]
    end = move_uci[2:4]

    start_pos = encode_square(start)
    end_pos = encode_square(end)

    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # if move is a promotion, find the piece
    # promoing to queen treated as default
    if len(move_uci) > 4:   
        p = move_uci[4]
        if p != 'q' and p != 'Q':
            return (dx, dy, p)

    return (dx, dy)


def legal_move_mask(board):
    move_mask = np.zeros((73, 8, 8))
    moves = board.legal_moves

    for m in moves:
        movestring = str(m)
        key = encode_move(movestring)
        pos = encode_square(movestring[:2])

        ind = MOVE_DICT[key]

        move_mask[ind, pos[0], pos[1]] = 1

    return torch.flatten(move_mask)


# you will likely never need to use this
def generate_move_dict():
    # 56 regular moves + 8 knight moves + 9 underpromotions

    move_dict = {}  # find move index from move
    moves = []

    for i in range(8):
        for j in range(7):
            d = DIRECTIONS[i]
            dx = d[0]*(j+1)
            dy = d[1]*(j+1)

            move = (dx, dy)

            move_dict[move] = 7*i + j
            moves.append(move)

    for i in range(8):
        move = KNIGHT_MOVES[i]

        move_dict[move] = 56 + i
        moves.append(move)

    for i in range(9):
        move = UNDERPROMOTIONS[i]

        move_dict[move] = 64 + i
        moves.append(move)

    return move_dict, moves


# this is specific to the dataset used
# https://www.kaggle.com/milesh1/35-million-chess-games
def move_list(game_string):
    moves = []
    recording = False

    s = ''

    for i in range(len(game_string)):
        char  = game_string[i]

        if char == '.':
            s = ''
            recording = True
            continue

        elif char == ' ':
            recording = False
            if s == '':
                continue

            moves.append(s)
            continue

        if recording:
            s = s + char

    return moves


def board_to_array(board):
    # encode board into neural network readable format

    piece_boards = []

    for i in range(2):
        for j in range(1, 7):
            piece_board = np.zeros((8, 8))

            positions = list(board.pieces(j, bool(i)))

            for p in positions:
                i, j = SQUARES[p]
                piece_board[i, j] = 1.0

            piece_boards.append(piece_board)

    piece_boards = np.array(piece_boards)

    turn = board.turn
    turn_board = np.ones((1, 8, 8)) if turn else np.zeros((1, 8, 8))

    piece_boards = np.concatenate((piece_boards, turn_board))

    for i in range(2):
        qcr = board.has_queenside_castling_rights(bool(i))
        kcr = board.has_kingside_castling_rights(bool(i))

        qcb = np.ones((1, 8, 8)) if qcr else np.zeros((1, 8, 8))
        kcb = np.ones((1, 8, 8)) if kcr else np.zeros((1, 8, 8))

        piece_boards = np.concatenate((piece_boards, qcb, kcb))

    ep_board = np.zeros((1, 8, 8))

    if board.ep_square:
        x, y = SQUARES[board.ep_square]
        ep_board[0, x, y] = 1.0

    threefold_repetition = board.can_claim_threefold_repetition()
    threefold_rep_board = np.ones((1, 8, 8)) if threefold_repetition else np.zeros((1, 8, 8))

    fifty_move_board = np.full((1, 8, 8), board.halfmove_clock / 100.0)

    piece_boards = np.concatenate((piece_boards, ep_board, threefold_rep_board, fifty_move_board))

    return piece_boards


def policy_index_to_move_tuple(index):
    a = index % 64

    x = (index - a) // 64

    z = a % 8
    y = (a - z) // 8

    return x, y, z

def move_tuple_to_policy_index(x, y, z):
    a = 8*y + z
    index = 64*x + a

    return index


# takes a game string and parses it to get pack sequence of board and moves to a .npz file
def game_data_to_npz(output_path, game_string, result):

    boards, moves, score = game_to_tuple(game_string, result)

    np.savez(output_path, boards=boards, moves=moves, scores=score)


# "tuple" here means decomposing into (boards, moves, scores)
def game_to_tuple(game_string, score):
    moves = move_list(game_string)
    board = chess.Board()

    encoded_moves = []
    encoded_boards = []
    scores = []

    for m in moves:
        move = board.push_san(m)

        # move encoding
        move_str = str(move)

        move_index = MOVE_DICT[encode_move(move_str)]
        sq = encode_square(move_str[:2])

        '''move_tensor = np.zeros((73, 8, 8))
        move_tensor[move_index, sq[0], sq[1]] = 1.0

        encoded_moves.append(move_tensor.flatten())'''

        ind = move_tuple_to_policy_index(move_index, sq[0], sq[1])

        encoded_moves.append(ind)

        # board encoding
        encoded_board = board_to_array(board)
        encoded_boards.append(encoded_board)

        scores.append(score)


    encoded_moves = np.array(encoded_moves)
    encoded_boards = np.array(encoded_boards)
    scores = np.array(scores)

    return encoded_boards, encoded_moves, scores
