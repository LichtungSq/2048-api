import numpy as np
from .displays import Display
from .game import Game
import keras
import os
from keras.models import load_model
from keras.utils import np_utils
# from keras.utils import load_model

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i: i for i in range(1,CAND)}
map_table[0] = 0

def grid_ohe(arr):
    ret = np.zeros(shape = OUT_SHAPE + (CAND,), dtype = int)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]] = 1
    return ret

class Agent(object):
    '''Agent Base.'''

    def __init__(self, game, display=None):
        #display = Display()
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        # display = Display()
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super(ExpectiMaxAgent, self).__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    
    def __init__(self, game,display = None):
        super(MyOwnAgent, self).__init__(game, display)
        self.model = load_model("./test_model_1352.h5")

    def step(self):
        board = np.array([grid_ohe(self.game.board)])
        tmp = self.model.predict(board)
        direction = int(tmp.argmax())
        return direction

# print(self.game.board)
# self.trainData.append(grid_ohe(self.game.board))
# self.trainLabel.append(direction_table[direction])
# TTAINDATA.append(grid_ohe(self.game.board))
