import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop, Adam 

from matplotlib import pyplot as plt 
import numpy as np
#Function
def Funti(x):
    return (2*np.cos(x)-(2/x)*np.sin(x)+x*np.sin(x))

"""
Inheritance of a class.  A class is modified from one already defined, 
in this case the "Sequential " class. Super" is used to say that it is 
a subclass of the original one. 

"""
class ODEsolver(Sequential): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
#You can change loss.tracker to one thing. 
    @property
    def metrics(self): #No changes metrics
        return [self.loss_tracker]

    """
    Database for neural network training. 
    A list of random values is 
    created in interval with a data size "batch_size". 
    """
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval= -5, maxval=5) #Changes the intervals


        with tf.GradientTape() as tape: #Calculate the gradients of weights and bias.
            # compute the loss value 

            with tf.GradientTape() as tape2: #Calculate the second derivative of x
                tape2.watch(x) #Monitors the operations made with x.
                y_pred = self(x, training=True) #evalua en x


            dy = tape2.gradient(y_pred, x) #deriva lo que predice la red con respecto a x
            x_o = tf.zeros((batch_size,1)) #condicion inical llena de zeros 
            y_o = self(x_o, training=True) #evalua la red en el tensor lleno de ceros
            eq = x*dy + y_pred - x**2*tf.math.cos(x)
            ic = y_o  #condicion incial deseada 
            loss = keras.losses.mean_squared_error(0., eq) + \
                keras.losses.mean_squared_error(0., ic) #Loss function

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        #mover en direccion contraria a los gradientes 
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss) #ver la funcion de costo
        #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

"""
Design of the dense neural network structure. 
"""
model = ODEsolver()

model.add(Dense(50, activation='tanh', input_shape=(1,)))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(optimizer=RMSprop(learning_rate=0.001), metrics=['loss'])

"""
Neural network testing.
Plot of analytical solution vs. the solution predicted by the network.
"""

NData = 300
minV = -5
maxV = 5
x = tf.linspace(minV, maxV, NData)
history = model.fit(x, epochs=500, verbose=1)


x_testv = x
a= model.predict(x_testv)
plt.plot(x_testv, a, label='Approximation NN')
plt.plot(x_testv, Funti(x), color= 'red', label='Analytical')
plt.legend()
plt.title('Gráfica ' + r'$2*Cos(x)-(2/x)*Sin(x)+x*Sin(x)$')
plt.show()
exit()
