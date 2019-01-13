import numpy as np
from .displays import Display
from .game import Game
import keras
import os
from keras.models import load_model
from keras.utils import np_utils
# from keras.utils import load_model

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        # display = Display()
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
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    
    def __init__(self,game,display = None):
        super(MyOwnAgent, self).__init__(game, display)
        self.model = load_model("./model_1300.h5")

    def step(self,game):
        direction = int(self.model.predict(self.game.board).argmax())
        return direction

# print(self.game.board)
# self.trainData.append(grid_ohe(self.game.board))
# self.trainLabel.append(direction_table[direction])
# TTAINDATA.append(grid_ohe(self.game.board))
