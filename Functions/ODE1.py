import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop, Adam 
import numpy as np 
#Library for graphing.
import matplotlib.pyplot as plt 
"""
Solución de la ODE: y''[x] + y[x]=0 y'[0]=-0.5 y[0]=1
"""
def Solve(x):
    return (1*tf.math.cos(x)-0.5*tf.math.sin(x))
"""
Inheritance of a class.  A class is modified from one already defined, 
in this case the "Sequential " class. Super" is used to say that it is 
a subclass of the original one. 
"""

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
        x_0 =tf.zeros((batch_size,1))


        with tf.GradientTape() as tape: #calcular los gradientes de pesos y bias
            # compute the loss value 
            with tf.GradientTape(persistent=True) as g:
                g.watch(x) #vigila las operaciones que se hagan con x
                #tape2.watch(x0)
                with tf.GradientTape(persistent=True) as gg:
                    gg.watch(x)
                    gg.watch(x_0)
                    y_pred = self(x, training=True) #evalua en x
                    y_0 = self(x_0, training=True)
                
                dy = gg.gradient(y_pred, x) #deriva lo que predice la red con respecto a x
                dy_0 = gg.gradient(y_0, x_0)

                #x_o =tf.zeros((batch_size,1))
                #y_o = self(x_o, training=True)
                ic = dy_0 + 0.5 # Condición y'[0]=-0.5
            
            dy2 = g.gradient(dy, x) #valor de la segunda derivada
            eq = dy2 + y_pred #Ecuación Diferencial a minimizar a cero
            #condicion incial deseada 

            x_o1 = tf.zeros((batch_size,1)) #condicion inical llena de zeros  
            y_o1 = self(x_o1, training=True) #evalua la red en esa condicion 
            ic1 = y_o1 - 1. # Condición y[0]=1
            
            
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic1) + keras.losses.mean_squared_error(0., ic)

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
#print(a)
plt.plot(x_testv, a, color='yellowgreen', label='Predict')
plt.plot(x_testv, Solve(x), color = 'violet', label='Actual')
plt.title('Gráfica ' + r'$Cos(x)-0.5*Sin(x)$')
plt.legend()
plt.show()
