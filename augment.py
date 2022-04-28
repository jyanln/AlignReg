import tensorflow.image as tf_image
import torchvision.transforms.functional as torch_transforms

# tensorflow
def tf_rotate(img):
    return tf_image.rot90(img)

def tf_flip(img):
    return tf_image.flip_left_right(img)

def tf_saturation(img):
    return tf_image.adjust_saturation(img, 3)

tf_default_augmentations = [tf_rotate, tf_flip]

# torch
def torch_rotate(img):
    return torch_transforms.rotate(img, 90)

def torch_flip(img):
    return torch_transforms.hflip(img)

torch_default_augmentations = [torch_rotate, torch_flip]
