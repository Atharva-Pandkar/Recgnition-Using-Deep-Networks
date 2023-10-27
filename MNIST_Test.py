# import statements
import Helper

import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# set output precision to 2
import Networks

torch.set_printoptions(precision=2)

'''
Load the model trained and saved by MNISTRecognition.py
Load the MNIST test data
Get the first ten test image and their predictions, plot the results
Load the custom 0 - 9 digits, apply the trained model
Plot the custom digits and their predictions
'''


def main(argv):
    """
    Task 1 F
    load the model
    """
    model = Networks.MNIST_Network()
    model.load_state_dict(torch.load('results\\model_state_dict.pth'))
    model.eval()

    # load test data
    test_loader = DataLoader(
        torchvision.datasets.MNIST('M_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    # get the label of the first ten images and print out the outputs
    first_ten_data, first_ten_label = Helper.first_n_output(test_loader, model,10)

    # plot the predictions for the first ten images
    import matplotlib.pyplot as plt
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(first_ten_data[i], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(first_ten_label[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # load custom digit data, apply the model, and plot the ten results
    image_dir = 'C:\\Users\\athar\\PycharmProjects\\pythonProject\\Assingment5\\Custom_number'
    custom_images = datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))]))
    custom_loader = DataLoader(custom_images)

    first_ten_custom_data, first_ten_custom_label = Helper.first_n_output(custom_loader, model, 10)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(first_ten_custom_data[i], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(first_ten_custom_label[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv)
