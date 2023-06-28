import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model')
img = np.load('ds.npy')[0][0].reshape(1, 50, 50, 1)

noise = np.random.uniform(0, 1, (1, 50, 50, 1))
# noise = img
for i in range(50):
    noise = model(noise)
    generated = np.array(noise).reshape(50, 50)
    plt.imshow(generated, cmap='gray')
    plt.show()