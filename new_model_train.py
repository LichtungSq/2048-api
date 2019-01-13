from game2048.game import Game
from game2048.expectimax import board_to_move
import numpy as np
import random
import os
import keras
from keras.models import *
from keras.layers import *
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# from keras.models import Sequential
# from keras.layers import concatenate, Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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

Guide = namedtuple('Guide', ('state', 'action'))

class Guides:

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        "Saves a transition."
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position+1)%self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def ready(self,batch_size):
        return len(self.memory) >= batch_size

    def _len_(self):
        return len(self.memory)

class ModelWrapper:
    
    def __init__(self,model,capacity):
        self.model = model
        self.memory = Guides(capacity)
        # self.writer = SummaryWriter()
        self.training_step = 0

    def predict(self,board):
        return model.predict(np.expand_dims(board,axis=0))

    def move(self,game):
        ohe_board = grid_ohe(game.board)
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)
        # print(game.score)

    def train(self,batch):
        if self.memory.ready(batch):
            guides = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss,acc = self.model.train_on_batch(np.array(X),np.array(Y))
            print('loss',float(loss),self.training_step)
            print('acc',float(acc),self.training_step)
            self.training_step += 1

def My_New_Model():
    inputs = Input((width,height,depth))

    conv1 = Conv2D(num_filters, (2, 1), kernel_initializer='he_uniform', padding='same')(inputs)
    conv2 = Conv2D(num_filters, (1, 2), kernel_initializer='he_uniform', padding='same')(inputs)
    conv11 = Conv2D(num_filters, (2, 1), kernel_initializer='he_uniform', padding='same')(inputs)
    conv12 = Conv2D(num_filters, (1, 2), kernel_initializer='he_uniform', padding='same')(inputs)
    conv21 = Conv2D(num_filters, (2, 1), kernel_initializer='he_uniform', padding='same')(inputs)
    conv22 = Conv2D(num_filters, (1, 2), kernel_initializer='he_uniform', padding='same')(inputs)
    
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv11 = LeakyReLU(alpha=0.3)(conv11)
    conv12 = LeakyReLU(alpha=0.3)(conv12)
    conv21 = LeakyReLU(alpha=0.3)(conv21)
    conv22 = LeakyReLU(alpha=0.3)(conv22)
    
    conv1 = MaxPooling2D()(conv1)
    conv2 = MaxPooling2D()(conv2)
    conv11 = MaxPooling2D()(conv11)
    conv12 = MaxPooling2D()(conv12)
    conv21 = MaxPooling2D()(conv21)
    conv22 = MaxPooling2D()(conv22)
    
    hidden = concatenate([Flatten()(conv1), Flatten()(conv2), Flatten()(conv11),
        Flatten()(conv12), Flatten()(conv21), Flatten()(conv22)])
    
    x = BatchNormalization()(hidden)
    x = Activation('relu')(x)
    
    for width in [512, 256]:
        x = Dense(width, kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

def train(self, begin_score, end_score):
    if os.path.exists("model_%d_%d.h5" % (begin_score, end_score)):
	    print("model exists, and will be loaded.")
	    self.model = load_model("model_%d_%d.h5" % (begin_score, end_score))
    else:
	    self.model = self.build()

# if os.path.exists("new_model.h5"):
model = load_model("new_model.h5")
# else:
#     model = My_New_Model()
task = ModelWrapper(model, 2**16)
batch = 2048
count = 0

while True:
    game = Game()
    count += 1
    while not game.end:
        task.move(game)
    print(game.score)
    task.train(batch = batch)

    if count % 10 == 0:
        model.save(filepath="mynew_model.h5", overwrite=True)

    if count % 100 == 0:
        model.save(filepath="mynew_model_100.h5", overwrite=True)
