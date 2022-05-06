import os
import sys
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

sys.path.append('../src')
from tf_training import *

batch_size = 128
imgDimension = (28, 28)
epochs = 3
input_size = (28, 28)


'loading the dataset'     

(ds_train, ds_test), ds_info = tfds.load('mnist',
                                         split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True,
                                         with_info=True
                                        )

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  print(image.shape)
  image = tf.image.resize(image, input_size)
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(batch_size)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=input_size),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation = 'softmax')
                                   ])


model2 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=input_size),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation = 'softmax')
                                   ])

keras.utils.plot_model(model, to_file='model_1.png', show_shapes=True)

optimizer=tf.keras.optimizers.Adam(0.001)
optimizer2=tf.keras.optimizers.Adam(0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss2 = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

l2_lambda = 0.005

'''
The imgMin and imgMax are also given default for the convinient of the user
So the user has to pass the following parameters into the train function
- train_data
- model
- optimizer
- epochs
- loss
- batch_size
- l2_lambda : The lambda to of l2 regularization
'''

acc, loss_hist = tf_train(ds_train, 
                        model, 
                        optimizer, 
                        epochs, 
                        loss,
                        l2_lambda)

acc2, loss_hist2 = tf_train(ds_train, 
                        model2, 
                        optimizer2, 
                        epochs, 
                        loss2,
                        0.0)

print(f"The length of the loss_history is {len(loss_hist)}")
fig, ax = plt.subplots(2)
ax[0].plot(loss_hist, color='r')
ax[1].plot(loss_hist2, color='b')
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

fig, ax = plt.subplots(2)
ax[0].plot(acc, color='r')
ax[1].plot(acc2, color='b')
plt.xlabel('Batch#')
plt.ylabel('Accuracy')
plt.show()
