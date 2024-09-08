import tensorflow as tf
import numpy as np

x_data=np.random.rand(100,1)
y_data=3 * x_data + 2 + np.random.randn(100,1)/1.5

model=tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_data,y_data,epochs=50)

predictions=model.predict(x_data)

print(predictions)

