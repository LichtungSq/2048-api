'''
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class model:
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

class ModelWrapper:
    
    def _init_(self,model,capacity):
        self.model = model
        self.memory = Guides(capacity)
        self.writer = SummaryWriter()
        self.training_step = 0

    def predict(self,board):
        return model.predict(np.expand_dims(board,axis=0))

    def move(self,game):
        ohe_board = game.ohe_board
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)

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
            self.writer.add_scalar('loss',float(loss),self.training_step)
            self.writer.add_scalar('acc',float(acc),self.training_step)
            self.training_step += 1
'''
