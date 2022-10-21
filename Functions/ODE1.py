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
        x = tf.random.uniform((batch_size, 1), minval= -5, maxval=5)
        x0 = tf.random.uniform((batch_size, 1), minval= -5, maxval=5)


        with tf.GradientTape() as tape: #calcular los gradientes de pesos y bias
            # compute the loss value 
            with tf.GradientTape(persistent=True) as g:
                g.watch(x) #vigila las operaciones que se hagan con x
                #tape2.watch(x0)
                with tf.GradientTape() as gg:
                    gg.watch(x)
                    y_pred = self(x, training=True) #evalua en x
                
                dy = gg.gradient(y_pred, x) #deriva lo que predice la red con respecto a x
                
                #x_o =tf.zeros((batch_size,1))
                #y_o = self(x_o, training=True)
                #ic = y_o + 0.5
            
            dy2 = g.gradient(dy, x) #valor de la segunda derivada
            eq = dy2 + y_pred #Ecuación Diferencial a minimizar a cero
            #condicion incial deseada 

            x_o1 = tf.zeros((batch_size,1)) #condicion inical llena de zeros  
            y_o1 = self(x_o1, training=True) #evalua la red en esa condicion 
            ic1 = y_o1 - 1.
            
            
            
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic1) #+ keras.losses.mean_squared_error(0., ic1)

        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        #mover en direccion contraria a los gradientes 
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
        self.loss_tracker.update_state(loss) #ver la funcion de costo
        #Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(50, activation='tanh', input_shape=(1,)))
#model.add(Dropout(0.2))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=Adam(),metrics=['loss']) #RMSprop(learning_rate=0.01)

dat=250
x = tf.linspace(-5,5,dat)
history = model.fit(x, epochs=500, verbose=1)



x_testv = tf.linspace(-5,5,dat)
a= model.predict(x_testv)
plt.plot(x_testv, a, label='Predict')
plt.plot(x_testv, (1*tf.math.cos(x)+0*tf.math.sin(x)), label='Actual')
plt.legend()
plt.show()
