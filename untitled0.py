# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yikYKHB9g3AkzmuyioFTV_Uk3RwPFd_Z
"""

import tensorflow as tf
import numpy as np

doble= np.array ([20,10,32,2],dtype=int)
mitad=np.array([10,5,16,1],dtype=int)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo= tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1),
    loss='mean_squared_error'
)

print("comenzando wowowowow")
historial= modelo.fit(doble, mitad,epochs=10000,verbose=False)
print("se acabo")

import matplotlib.pyplot as plt
plt.xlabel('#edad')
plt.ylabel('fallo')
plt.plot(historial.history["loss"])

print("predeiccion!!!")
resultado=modelo.predict([-4564545])
print("la mitad es:"+str(resultado))