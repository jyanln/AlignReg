import os
import time 
from torchattacks import PGD
from scipy.stats import wasserstein_distance
from PIL import Image
import cv2
from tqdm import tqdm
from pprint import pprint
from sklearn.utils import shuffle
import numpy as np
from matplotlib import pyplot as plt
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
# from torchvision import transforms, datasets

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
# tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()
# tf.executing_eagerly()