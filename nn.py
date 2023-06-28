import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Reshape
import numpy as np

print('different')

tds = np.load('ds.npy')
train_x = np.array([item[0] for item in tds]).astype(np.float32).reshape(-1, 50, 50, 1)
train_y = np.array([item[1] for item in tds]).astype(np.float32).reshape(-1, 50, 50, 1)
pipeline = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(8)


img_input = Input(shape=(50, 50, 1))
x = Conv2D(8, kernel_size=(2, 2), activation='relu', padding='same')(img_input)
x = MaxPooling2D(pool_size=1, padding='same')(x)

x = Conv2D(8, kernel_size=(2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=1, padding='same')(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(50*50)(x)
output = Reshape(target_shape=(50, 50, 1))(x)


model = tf.keras.Model(img_input, output)
model.compile(optimizer='adam', loss='mse')
model.summary()

print(train_x.shape, train_y.shape)
try:
    model.fit(pipeline, epochs=1)
except KeyboardInterrupt:
    model.save('model')
model.save('model')
