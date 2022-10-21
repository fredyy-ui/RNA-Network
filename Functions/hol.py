
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import RMSprop, Adam 

from matplotlib import pyplot as plt 
import numpy as np

import numpy as np

def y(x):
    return (np.sin(2*np.pi*x) + np.sin(5*np.pi*x))

x_vals = np.arange(-1,1,0.01)
y_vals = y(x_vals)
print(y_vals)
print('------')
y_max = y_vals.max()
y_vals /= y_max

print(y_max)
print(y_vals)

x_o = tf.zeros((10,1)) #condicion inical llena de zeros
print(x_o)
x_oo = x_o + 3

print(x_oo)

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-np.pi/2, np.pi/2, 31)
y = np.cos(x)**3

# 1) remove points where y > 0.7
x2 = x[y <= 0.7]
y2 = y[y <= 0.7]

# 2) mask points where y > 0.7
y3 = np.ma.masked_where(y > 0.7, y)

# 3) set to NaN where y > 0.7
y4 = y.copy()
y4[y3 > 0.7] = np.nan

plt.plot(x*0.1, y, 'o-', color='lightgrey', label='No mask')
plt.plot(x2*0.4, y2, 'o-', label='Points removed')
plt.plot(x*0.7, y3, 'o-', label='Masked values')
plt.plot(x*1.0, y4, 'o-', label='NaN values')
plt.legend()
plt.title('Masked and NaN data')
plt.show()

x = tf.random.uniform((20, 1), minval= -2, maxval=2)
print(x)