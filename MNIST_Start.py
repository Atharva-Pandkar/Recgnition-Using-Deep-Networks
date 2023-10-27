# import statements
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import Helper
import Networks

'''
Task 1 A
Load the MNIST training and testing dataset
Plot the first 6 images in the training dataset
Train and test the model, plot the training curve
Save the model and its state dict
'''

# main function
def main(argv):
    # Task 1 B make the network code repeatable
    random_seed = 28
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # load test and training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('M_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('M_Data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    # plot the first 6 example digits
    Helper.plot_images(train_loader, 2, 3)

    # initialize the network and the optimizer
    network = Networks.MNIST_Network()
    optimizer = optim.SGD(network.parameters(), lr=Networks.LEARNING_RATE,
                          momentum=Networks.MOMENTUM)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(Networks.N_EPOCHS + 1)]

    '''
    Task 1 D
    run the training
    '''
    Helper.test(network, test_loader, test_losses)
    for epoch in range(1, Networks.N_EPOCHS + 1):
        Helper.train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        Helper.test(network, test_loader, test_losses)

    # plot training curve
    Helper.plot_curve(train_counter, train_losses, test_counter, test_losses)

    '''
    Task 1 E
    save the model
    '''
    torch.save(network, 'results\\model.pth')
    torch.save(network.state_dict(), 'results\\model_state_dict.pth')

    return


if __name__ == "__main__":
    main(sys.argv)