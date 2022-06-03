import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence
import torch

def tf_sql2(model, images, aug_images):
    logits_orig = model(images, training=False)
    logits_aug = model(aug_images, training=False)
    sq_dists = tf.math.square(tf.math.subtract(logits_orig, logits_aug))
    sql2 = tf.math.reduce_sum(sq_dists)
    return sql2

def tf_maxsql2(model, images, aug_image_sets, loss_fn, labels, l2_lambda):
    '''
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset
    aug_images: list of batches, where each batch has an agumentation applied
    loss_fn: loss function to be used
    labels: labels of the images
    l2_lambda: the lambda parameter for the sql2
    '''
    if len(aug_image_sets) > 0:
        maxloss_image = max(aug_image_sets, key=lambda aug: loss_fn(labels, model(aug, training=False)))
        maxloss = loss_fn(labels, model(maxloss_image, training=False))
        sql2 = tf_sql2(model, images, maxloss_image)
        return l2_lambda * (maxloss + sql2)
    else:
        return 0
    
def torch_sql2(model, image, aug_image):
    logits_orig = model(image)
    logits_aug = model(aug_image)
    sq_dists = torch.square(torch.sub(logits_orig, logits_aug))
    sql2 = torch.sum(sq_dists)
    return sql2

def torch_maxsql2(model, image, aug_images, loss_fn, labels, l2_lambda):
    '''
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset
    aug_images: list of batches, where each batch has an agumentation applied
    loss_fn: loss function to be used
    labels: labels of the images
    l2_lambda: the lambda parameter for the sql2
    '''
    if aug_images:
        maxloss_image = max(aug_images, key=lambda aug: loss_fn(model(aug), labels))
        maxloss = loss_fn(model(maxloss_image), labels)
        sql2 = torch_sql2(model, image, maxloss_image)
        return l2_lambda * (maxloss + sql2)
    else:
        return 0

