import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

import numpy as np
np.set_printoptions(linewidth=200)

import matplotlib.pyplot as plt
from datetime import datetime

# Load the TensorBoard notebook extension.
#%load_ext tensorboard

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

random_index = np.random.randint(0,59999)
plt.imshow(training_images[random_index])
print(training_labels[random_index])
print(training_images[random_index])

training_images  = training_images / 255.0
test_images = test_images / 255.0


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])





model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])



          
          
model.summary()

print("Average training loss: ", np.average(training_history.history['loss']))
print("Average training accuracy: ", np.average(training_history.history['accuracy']))

# Write scores to a file
#with open("metrics.txt", 'w') as outfile:
#        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
#        outfile.write("Test variance explained: %2.1f%%\n" % test_score)
