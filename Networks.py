"""Import Statements"""
from torch import nn
import torch.nn.functional as F

'''define hyper-parameters'''
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10

''' 
Task 1 C
                model definition
A convolution layer with 10 5x5 filters
A max pooling layer with a 2x2 window and a ReLU function applied.
A convolution layer with 20 5x5 filters
A dropout layer with a 0.5 dropout rate (50%)
A max pooling layer with a 2x2 window and a ReLU function applied
A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
'''


class MNIST_Network(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # compute a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        x = x.view(-1, 320)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 50 nodes
        x = self.fc2(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, 1)  # apply log_softmax()


