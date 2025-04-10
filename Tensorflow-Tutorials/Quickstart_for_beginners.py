import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
import numpy as np

print("TensorFlow Version :", tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(10)
])

#predictions = model(x_train[:1]).numpy()

#tf.nn.softmax(predictions).numpy()

#loss_fn = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)

#loss_fn(y_train[:1], predictions).numpy()

#model.compile(optimizer='adam', loss=loss_fn, metrics = ['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)