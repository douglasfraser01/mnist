import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

import numpy as np
np.set_printoptions(linewidth=200)

import matplotlib.pyplot as plt
from datetime import datetime

# Load the TensorBoard notebook extension.
%load_ext tensorboard

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

random_index = np.random.randint(0,59999)
plt.imshow(training_images[random_index])
print(training_labels[random_index])
print(training_images[random_index])

training_images  = training_images / 255.0
test_images = test_images / 255.0

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


training_history = model.fit(training_images, 
          training_labels, 
          #batch_size = 64,
          #verbose = 0,
          epochs = 5,
          #validation_data = (test_images, test_labels), 
          callbacks=[tensorboard_callback])
          
          
model.summary()
