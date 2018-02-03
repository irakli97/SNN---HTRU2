import numpy as np
from sklearn import preprocessing
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout
from keras.utils  import to_categorical
from keras import backend as K

K.set_image_dim_ordering('th')
np.random.seed(7) 
my_data = np.array(pd.read_csv('HTRU_2.csv', sep=',',header=None))

x = my_data[:,0:8]
y = my_data[:,8]

y = to_categorical(y)
num_classes = 2
x = preprocessing.scale(x)
#x = (x - np.mean(x)) / np.std(x) 

model = Sequential()
'''
Intent:
    Conic layer form
    8 hidden layers starting with 512 units. 
    AlphaDropout with rate = 0.05
    inputs normalized to zero mean and unit variance   
    
'''
model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal', input_shape=(8,)))
model.add(AlphaDropout(0.05))
model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(128, activation='selu', kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(64, activation='selu',  kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(32, activation='selu',  kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(16, activation='selu',  kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(8, activation='selu',   kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))
model.add(Dense(4, activation='selu',   kernel_initializer='lecun_normal'))
model.add(AlphaDropout(0.05))


#output Layer
model.add(Dense(2, activation='softmax'))
# Compile model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=0.01, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(x, y, validation_split=0.3, epochs=10, batch_size=100, verbose=1, shuffle=True)
# Final evaluation of the model
scores = model.evaluate(x, y, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
# Error: 2.03%
