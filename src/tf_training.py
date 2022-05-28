import tensorflow as tf
import tensorflow.keras as keras
from regularization import tf_maxsql2
from augment import tf_default_augmentations
from tqdm import tqdm
import numpy as np

accuracy_metric = keras.metrics.Accuracy()
loss_history = []
accuracy = []

def measure_acc(model, images, labels, accuracy_metric, test_data = None):
    '''
    The function provides the accuracy of the model given the parameters.
    
    Attrs:
    -----
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset, where no_of_images = batch_size
    labels: labels of the images came in a iteration of batch_size
    accuracy_metric: The accuracy_metric defined above in the 1st line of the cell
    '''  

    # If test data is available, we use that to measure accuracy
    if test_data:
        accuracy_metric.reset_state()
        for (batch, (images, labels)) in enumerate(test_data):
            prediction = model(images, training = True)
            pred_y = tf.argmax(prediction, 1)
            accuracy_metric.update_state(pred_y, labels)
        return accuracy_metric.result().numpy()
    else:
        prediction = model(images, training = True)
        pred_y = tf.argmax(prediction, 1)
        accuracy_metric.update_state(pred_y, labels)
        return accuracy_metric.result().numpy()

def train_step(model, images, aug_images, labels, optimizer, loss_fn, l2_lambda, test_data = None):
    '''
    Attrs:
    -----
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset, where no_of_images = batch_size
    aug_images: augmeted all images cam in a batch using the augment_3 function in augment cell  
    labels: labels of the images
    optimizer: optimizer used by the model while training 
    loss_fn: loss function used by the model while training
    l2_lambda: The lambda parameter for applying the l2_regularization
    '''
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_val = loss_fn(labels, logits)
        loss_val += sum(model.losses)

        # squared l2 regularization
        loss_val += tf_maxsql2(model, images, aug_images, loss_fn, labels, l2_lambda)

    loss_history.append(loss_val)

    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val

def tf_train(train_data, 
             test_data,
             model, 
             optimizer, 
             epochs, 
             loss_fn, 
             l2_lambda,
             augmentations = tf_default_augmentations,
             lazy_augmentation = False):
    '''
    The function train the model if written by user in Tensorflow
    and return the loss and the accuracy of the model given the 
    parameters
    
    Attrs:
    -----
    train_data: intializer of the data_generator defined by the user 
    model: the model passed defined by the user for training
    optimizer: optimizer used by the model while training 
    loss_fn: loss function used by the model while training
    l2_lambda: The lambda parameter for applying the l2_regularization
    lazy_augmentation: Whether to augment images as they are used, as opposed
                       to preprocessing every augmented image
    test_data: Data to be used for testing
    '''
    accuracy.clear()
    loss_history.clear()
    accuracy_metric.reset_state()

    # preprocess by augmenting the images
    if not lazy_augmentation:
        aug_data = train_data.map(lambda img, label: (img, list(map(lambda aug: aug(img), augmentations)), label))
    else:
        aug_data = train_data.map(lambda img, label: (img, [], label))

    # training
    for epoch in tqdm(range(epochs)):
        for (batch, (images, aug_images, labels)) in enumerate(aug_data):
            if lazy_augmentation:
                aug_images = list(map(lambda aug: aug(images), augmentations))

            train_step(model, images, aug_images, labels, optimizer, loss_fn, l2_lambda, test_data)
        
        accuracy.append(measure_acc(model, images, labels, accuracy_metric, test_data))

    return accuracy[:], loss_history[:]
