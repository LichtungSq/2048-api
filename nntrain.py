from game2048.game import Game
from game2048.expectimax import board_to_move
import numpy as np
import keras

def grid_ohe(arr):
    ret = np.zeros(shape = OUT_SHAPE + (CAND,), dtype = int)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]] = 1
    return ret

class Guides:

    def _init_(self,capacity):
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
        return len(self.memory) >= batch.size

    def _len_(self):
        return len(self.memory)

class ModelWrapper:
    
    def _init_(self,model,capacity):
        self.model = model
        self.memory = Guides(capacity)
        self.writer = SummaryWriter()
        self.training_step = 0

    def predict(self,board):
        return model.predict(np.expand_dims(board,axis=0))

    def move(self,game):
        ohe_board = grid_ohe(game.board)
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)
        print(game.score)

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

class Model:

    inputs = Input((4,4,16))

    conv = inputs
    FILTERS = 128
    conv41 = Conv2D(filters=FILTERS,kernel_size=(4,1),kernel_initializer ='he_uniform')(conv)
    conv14 = Conv2D(filters=FILTERS,kernel_size=(1,4),kernel_initializer ='he_uniform')(conv)
    conv22 = Conv2D(filters=FILTERS,kernel_size=(2,2),kernel_initializer ='he_uniform')(conv)
    conv33 = Conv2D(filters=FILTERS,kernel_size=(3,3),kernel_initializer ='he_uniform')(conv)
    conv44 = Conv2D(filters=FILTERS,kernel_size=(4,4),kernel_initializer ='he_uniform')(conv)

    hidden = concatenate([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
    x = BatchNormalization()(hidden)
    x = Activation('relu')(hidden)

    for width in [512,128]:
        x = Dense(width,kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    outputs = Dense(4,activation = 'softmax')(x)
    model = Model(inputs,outputs)
    model.summary()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model = Model()
mw = ModelWrapper(model,2**16)
batch = 2048

while True:
    game = Game()
    while not game.end:
        mw.move(game)
    mw.train(batch = batch)
    if :
        model.save("dfd")
    if 1000:
        model.save()
        


