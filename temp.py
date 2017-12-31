import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
numpy.random.seed(7) 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')  

X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
model = Sequential()
'''
Intent:
    Conic layer form
    8 hidden layers starting with 256 units. 
    2^8 = 256
    AlphaDropout with rate = 0.05
    inputs normalized to zero mean and unit variance, NOT SURE    
    
'''
model.add(Flatten(input_shape=(1, 28, 28)))

model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(128, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(64, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(32, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(16, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
'''
model.add(Dense(8, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(4, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(2, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
        #output Layer
model.add(Dense(1, activation='softmax'))

'''

model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))


