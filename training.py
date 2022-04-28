import os
import sys
# import code/regularization_config/config.py



all_paths = ['code/augmentation',
             'code/regularization_config',
             'code/tf_and_torch_train']

'''
for colab 
'''
for i in range(len(all_paths)):
    sys.path.append(os.path.join('/content/boraintation', all_paths[i]))

'''             
for i in range(len(all_paths)):
    sys.path.append(all_paths[i])
'''

from torch_training import torch_train
from tf_training import tf_train

def train(train_data, 
          model, 
          optimizer, 
          epochs, 
          loss,
          l2_lambda):
    '''
    It identifies the which function to call, i.e. torch_train
    or tf_train based on the model provided by the user. 

    Attrs:
    -----
    train_data: intializer of the data_generator defined by the user 
    model: the model passed defined by the user for training
    optimizer: optimizer used by the model while training 
    loss: loss function used by the model while training
    l2_lambda: The lambda parameter for applying the l2_regularization
    '''  

    if list(model.__dict__.keys())[0] == 'training':
        accuracy, loss_history = torch_train(train_data, model, optimizer, epochs, loss, l2_lambda)
    elif list(model.__dict__.keys())[0] == '_self_setattr_tracking':
        accuracy, loss_history = tf_train(train_data, model, optimizer, epochs, loss, l2_lambda)


    return accuracy, loss_history