import os

import numpy as np 
import torch

from constants import *
from nn import *
from utils import *



def policy_vector_test():
    indices = (63, 64, 136, 932, 1999, 4559)

    for ind in indices:
        x, y, z = policy_index_to_move_tuple(ind)
        
        t = torch.zeros((1, 73, 8, 8))
        t[0, x, y, z] = 1

        t = torch.nn.Flatten()(t)

        assert t.shape[1] == 4672

        assert torch.argmax(t).item() == ind
        assert move_tuple_to_policy_index(x, y, z) == ind


def game_saving_loading_test():
    games, results = parse_raw_data(0, 6)

    game_data_to_npz('test_files\\test', games[0], results[0])

    assert os.path.exists('test_files\\test.npz')

    a = np.load('test_files\\test.npz')

    assert a['boards'].shape == (67, 20, 8, 8)
    assert a['moves'].shape == (67,)
    assert a['scores'].shape == (67,)


# TODO: finish
def test_promotions():
    b = chess.Board()
    moves = ['e4', 'd5', 'exd5', 'c6', 'd6', 'e6', 'd7', 'Ke7']  # promotion test
    # 'dxc8q'

    for m in moves:
        b.push_san(m)

    moves = []

    for m in b.legal_moves:
        moves.append(str(m))

    assert 'd7c8q' in moves
    assert 'e1e2' in moves
    assert 'h2h4' in moves

    b.push_san('dxc8q')




policy_vector_test()
game_saving_loading_test()
test_promotions