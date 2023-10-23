

import tensorflow as tf
import numpy as np

doble= np.array ([20,10,32,2],dtype=int)
mitad=np.array([10,5,16,1],dtype=int)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo= tf.keras.Sequential([capa])
#algoritmo de aprendizaje
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1),
    loss='mean_squared_error'
)

print("comenzando wowowowow")
historial= modelo.fit(doble, mitad,epochs=10000,verbose=False)
print("se acabo")
#print a graph of the evolution of the ia
import matplotlib.pyplot as plt
plt.xlabel('#edad')
plt.ylabel('fallo')
plt.plot(historial.history["loss"])

print("predeiccion!!!")
resultado=modelo.predict([6])#input
print("la mitad es:"+str(resultado))
