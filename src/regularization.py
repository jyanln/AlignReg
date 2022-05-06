import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import KLDivergence
import torch

def tf_sql2(model, images, aug_images, l2_lambda):
    logits_orig = model(images, training=False)
    logits_aug = model(aug_images, training=False)
    sq_dists = tf.math.square(tf.math.subtract(logits_orig, logits_aug))
    sql2 = tf.math.reduce_sum(tf.math.scalar_mul(l2_lambda, sq_dists))
    return sql2

def tf_maxsql2(model, images, aug_image_sets, loss_fn, labels, l2_lambda):
    '''
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset
    aug_images: list of batches, where each batch has an agumentation applied
    l2_lambda: the lambda parameter for applying the l2_regularization
    '''
    if len(aug_image_sets) > 0:
        maxloss = max(aug_image_sets, key=lambda aug: loss_fn(labels, model(aug, training=False)))
        return tf_sql2(model, images, maxloss, l2_lambda)
    else:
        return 0
    
def torch_sql2(model, image, aug_image, l2_lambda):
    logits_orig = model(image)
    logits_aug = model(aug_image)
    sq_dists = torch.square(torch.sub(logits_orig, logits_aug))
    sql2 = torch.sum(torch.mul(sq_dists, l2_lambda))
    return sql2

def torch_maxsql2(model, image, aug_images, loss_fn, labels, l2_lambda):
    '''
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset
    aug_images: list of batches, where each batch has an agumentation applied
    l2_lambda: the lambda parameter for applying the l2_regularization
    '''
    if aug_images:
        maxloss = max(aug_images, key=lambda aug: loss_fn(model(aug), labels))
        return torch_sql2(model, image, maxloss, l2_lambda)
    else:
        return 0

