import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import Helper
import Networks

batch_size_test = 64

# class definition

'''
A deep network takes different number of convolution layers, different filter sizes of the convolution layer, and different dropout rate
It contains the following layers:
A convolution layer with 10 5x5 filters
A max pooling layer with a 2x2 window and a ReLU function applied.
A convolution layer with 20 5x5 filters
A dropout layer (rate will be passed by parameter when initialize the model)
A max pooling layer with a 2x2 window and a ReLU function applied
Some convolution layers with different filter sizes(pass by parameter when initialize the model)
A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
'''
class ExperimentNetwork(nn.Module):
    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        super(ExperimentNetwork, self).__init__()
        self.input_size = 28 # input image size is 28x28
        self.num_of_conv = num_of_conv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    '''
    The function gets the input size for the first fully connected layer
    '''
    def get_fc1_input_size(self):
        fc1_input_size = self.input_size / 2
        fc1_input_size = fc1_input_size / 2
        fc1_input_size = fc1_input_size * fc1_input_size * 20
        return int(fc1_input_size)

    '''
    The function computes a forward path for the network
    '''
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_of_conv):
            x = F.relu(self.conv(x))
        # x = x.view(-1, )
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)


# helper functions

'''
The function loads training and test data, initializes a network, trains the network.
The function prints the model accuracy and plots the training and testing losses.
@:parameter num_epochs: number of epochs of the training process
@:parameter batch_size_train: the batch size of the traning data
@:parameter num_of_conv: the number of convolution layers in the model
@:parameter conv_filter_size: the filter size in the convolution layers
@:parameter dropout_rate: the dropout rate of the model
@:parameter filename: the path to save the plots of the training and testing losses
'''
def experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, filename):
    # load test and training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('experiment_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('experiment_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test)

    # initialize the network and the optimizer
    network = ExperimentNetwork(num_of_conv, conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=Networks.LEARNING_RATE,
                          momentum=Networks.MOMENTUM)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_epochs + 1)]

    # run the training
    Helper.test(network, test_loader, test_losses)
    for epoch in range(1, num_epochs + 1):
        Helper.train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        Helper.test(network, test_loader, test_losses)

    # plot training curve
    plot_curve(train_counter, train_losses, test_counter, test_losses, filename)


'''
The function plots curves of the training loses and testing losses
@:parameter train_counter: array of train counter
@:parameter train_losses: array of train losses
@:parameter test_counter: array of test counter
@:parameter test_losses: array of test losses
'''
def plot_curve(train_counter, train_losses, test_counter, test_losses, filename):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(filename)


# main function

'''
'''
def main():
    for num_epochs in [3, 5]:
        for batch_size_train in [64, 128]:
            for num_of_conv in range(1, 3):
                for conv_filter_size in [3, 5]:
                    for dropout_rate in [0.3, 0.5]:
                        filename = f'analysis\\epochs_{num_epochs}_batchSize_{batch_size_train}_numofConv_{num_of_conv}' \
                                   f'_convFilSize_{conv_filter_size}_Dropout_{dropout_rate}.png'
                        print('______________________________')
                        print(f'Number of Epochs: {num_epochs}')
                        print(f'Train Batch Size: {batch_size_train}')
                        print(f'Number of Convolution Layer: {num_of_conv}')
                        print(f'Convolution Filter Size: {conv_filter_size}')
                        print(f'Dropout Rate: {dropout_rate}')
                        print('______________________________')
                        experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, filename)


if __name__ == "__main__":
    main()