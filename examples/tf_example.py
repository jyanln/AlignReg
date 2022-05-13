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

# Setup tf model and training
batch_size = 128
imgDimension = (28, 28)
epochs = 3
input_size = (28, 28)

(ds_train, ds_test), ds_info = tfds.load('mnist',
                                         split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True,
                                         with_info=True
                                        )

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
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

keras.utils.plot_model(model, to_file='model_1.png', show_shapes=True)

optimizer=tf.keras.optimizers.Adam(0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Set regularization factor
l2_lambda = 0.01

# Set augmentation functions
# Here we use the default list from the augment module
augmentations = tf_default_augmentations

# Pass variables into training function
# We can choose to enable lazy_augmentations to augment the image batches as
# they are used. Otherwise, all augmentations will be preprocessed, which may
# be too large for the disk.
acc, loss_hist = tf_train(ds_train, 
                        model, 
                        optimizer, 
                        epochs, 
                        loss,
                        l2_lambda,
                        augmentations=tf_default_augmentations,
                        lazy_augmentation=True,
                        test_data=ds_test)

# Plot data output
plt.plot(loss_hist, color='r')
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

plt.plot(acc, color='r')
plt.xlabel('Batch#')
plt.ylabel('Accuracy')
plt.show()
