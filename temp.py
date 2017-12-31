import numpy
from sklearn import preprocessing
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import AlphaDropout
from keras.layers import Flatten
from keras.utils import np_utils, to_categorical
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th')
numpy.random.seed(7) 
my_data = numpy.array(pd.read_csv('HTRU_2.csv', sep=',',header=None))
x = my_data[:,[0,1,2,3,4,5,6,7]]
y=my_data[:,8]

y = to_categorical(y)
num_classes = 2



x = preprocessing.scale(x)



model = Sequential()
'''
Intent:
    Conic layer form
    8 hidden layers starting with 256 units. 
    2^8 = 256
    AlphaDropout with rate = 0.05
    inputs normalized to zero mean and unit variance, NOT SURE    
    
'''
model.add(Dense(512,activation='selu',kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
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

model.add(Dense(8, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(4, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(2, activation='selu', kernel_initializer='lecun_normal', bias_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))

#output Layer
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x, y, validation_split=0.3, epochs=100, batch_size=100, verbose=1, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))


