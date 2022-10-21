import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop


import numpy as np
from matplotlib import pyplot as plt

"""
I define the function to be
 approximated with an NN
""" 
def Funti(x):
    return ((1)+(2)*x+(4)*x*x*x)
"""
We create the database, x values in a given interval.
"""
NData=1200
minval = -1
maxval = 1
x_train = tf.random.uniform((NData, 1), minval, maxval)
y_train = Funti(x_train)

print(x_train)
print(y_train)

"""
Design of the neural network structure.
"""
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
#model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
"""
Neural network training.
"""
model.compile(loss='mse',optimizer=RMSprop(learning_rate=0.001)) #, metrics=['accuracy']
history = model.fit(x_train, y_train, batch_size=50, epochs=110, verbose=1)

"""
Test of the neural network and plot of the function with respect to 
the function predicted by the network. 

"""
x = tf.linspace(minval, maxval, NData)
print(x)
a = model.predict(x)
print(a)
plt.plot(x, a, color= 'red', label='Approximation NN')
plt.plot(x, (Funti(x)), color= 'yellow', label='Analytical')
plt.legend()
plt.title('Gráfica ' + r'$f(x)= 1+ 2x + 4 x^{3}$')
plt.show()