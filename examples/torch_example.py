import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

sys.path.append('../src')
from augment import *
from torch_training import *

# Setup torch model and training
batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainsetraw = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# We will use the AugmentedDataset class to specify the augmentations, which
# will be applied to the images automatically.
# Here we use the default list from the augment module
augmentations = torch_default_augmentations

# We can choose to enable lazy_augmentations to augment the image batches as
# they are used. Otherwise, all augmentations will be preprocessed, which may
# be too large for the disk.
trainset = AugmentedDataset(trainsetraw, augmentations, lazy_augmentation=True)

trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss = nn.CrossEntropyLoss()

epochs=3
l2_lambda = 0.01

# Pass variables into training function
accuracy, loss_history = torch_train(trainloader, testloader, net, optimizer, epochs, loss, l2_lambda, record_batch_size=64, acc_batch=2)

# Check final accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# Plot data output
plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

plt.plot(accuracy)
plt.xlabel('Batch#')
plt.ylabel('Accuracy')
plt.show()
