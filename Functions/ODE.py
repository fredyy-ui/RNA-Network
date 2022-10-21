import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop, Adam 

from matplotlib import pyplot as plt 
import numpy as np
#herencia de una clase, se modifica una clase ya hecha
#constructor superr
class ODEsolver(Sequential): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
#Puedo cambiar loss_tracker por otra cosa
    @property
    def metrics(self): #no cambiar metrics
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size, 1), minval= -2, maxval=2)


        with tf.GradientTape() as tape: #calcular los gradientes de pesos y bias
            # compute the loss value 
            with tf.GradientTape() as tape2:
                tape2.watch(x) #vigila las operaciones que se hagan con x
                y_pred = self(x, training=True) #evalua en x
            dy = tape2.gradient(y_pred, x) #deriva lo que predice la red con respecto a x
            #dy2 = tape2.gradient(dy, x)
            x_o = tf.zeros((batch_size,1)) #condicion inical llena de zeros
            #x_c1 = x_o + 3
            #x_c2 = x_o + 2 
            y_o = self(x_o, training=True) #evalua la red en esa condicion 
            #y_c1 = self(x_c1, training=True)
            #y_c2 = self(x_c2, training=True)
            eq = x*dy + y_pred - x**2*tf.cos(x)
            ic = y_o  #condicion incial deseada 
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        #mover en direccion contraria a los gradientes 
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss) #ver la funcion de costo
        #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(5, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

dat=300
x = tf.linspace(-5,5,dat)
history = model.fit(x, epochs=400, verbose=1)



x_testv = tf.linspace(-5,5,dat)
a= model.predict(x_testv)
plt.plot(x_testv, a, label='Predict')
plt.plot(x_testv, (2*np.cos(x)-(2/x)*np.sin(x)+x*np.sin(x)), label='Actual')
plt.legend()
plt.show()
exit()
