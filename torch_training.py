import numpy as np
import torch
from tqdm import tqdm
from regularization import torch_maxsql2
from augment import torch_default_augmentations

#TODO remove
import torchvision
import torchvision.transforms as transforms

loss_history = []
accuracy = []

def measure_acc(model, images, labels, batch_size):
    '''
    The function provides the accuracy of the model given the parameters.
    
    Attrs:
    -----
    model: the model passed defined by the user for training
    images: images came in a batch of the dataset, where no_of_images = batch_size
    labels: labels of the images came in a iteration of batch_size
    batch_size: batch_size
    '''

    prediction = model(images.clone().detach())
    pred_y = np.array(torch.argmax(prediction, 1))
    z = 0
    for (a,b) in zip(pred_y, np.array(labels)):
        if a == b:
            z+=1
    return z/batch_size

def torch_train(train_data, 
                model, 
                optimizer, 
                epochs, 
                loss_fn,
                l2_lambda, 
                augmentations = torch_default_augmentations):
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
    '''
    accuracy.clear()
    loss_history.clear()

    #TODO preprocess by augmenting the images
    #TODO implement batch size
    batch_size = 500

    # training
    for epoch in tqdm(range(epochs)):

        running_loss = 0.0
        for i, (image, label) in enumerate(train_data, 0):
            aug_images = map(lambda aug: aug(image), augmentations)

            optimizer.zero_grad()

            logits = model(image)
            loss_val = loss_fn(logits, label)


            reg = torch_maxsql2(model, image, aug_images, l2_lambda)
            loss_val = torch.add(loss_val, reg.item())
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()
            if i % batch_size == batch_size - 1:
                loss_history.append(running_loss)
                running_loss = 0.0
                accuracy.append(measure_acc(model, image, label, batch_size))

    return accuracy[:], loss_history[:]
