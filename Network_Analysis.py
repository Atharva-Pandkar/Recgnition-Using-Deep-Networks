# import statements
import Helper
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision

import Networks

'''
Load the model trained by MNISTRecognition.py
Plot the 10 filters in the first convolution layer of the model
Load the MNIST training data
Apply the 10 filters to the first image in the training dataset
Plot the 10 filters and 10 filtered images
Load a truncated model from the previous model
Apply the 20 filters from the second convolution layer of this model to the first image from the training dataset
Plot the 20 filters and the 20 filtered images
'''
def main(argv):

    # load and print the model
    model = Networks.MNIST_Network()
    model.load_state_dict(torch.load('results\\model_state_dict.pth'))
    print(model)

    '''
     Print the filter wights and shape for the first conv1 layer
     plot the 10 filters
    '''
    filters = plot_filters(model.conv1, 10, 3, 4)

    # apply the 10 filters to the first training example image
    # load training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('M_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))
    # Task 2 B
    #  get the first image

    first_image, first_label = next(iter(train_loader))
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    # plot the first images filtered by the 10 filters from layer 1
    plot_filtered_images(filters, squeezed_image, 10, 20, 5, 4)

'''
The function plots some filters
@:parameter conv: the convolutation layer from a model which contains the filters to be plotted
@:parameter total: total number of filters to be plotted
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot
@:return filters:; array of all the filters plotted
'''
def plot_filters(conv, total, row, col):
    filters = []
    with torch.no_grad():
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            curr_filter = conv.weight[i, 0]
            filters.append(curr_filter)
            print(f'filter {i + 1}')
            print(curr_filter)
            print(curr_filter.shape)
            print('\n')
            plt.imshow(curr_filter)
            plt.title(f'Filter {i + 1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    return filters


'''
The function plots filters and filtered images
@:parameter filters: the filters to be plotted
@:parameter image: the image to be filtered
@:parameter n: the total number of filters
@:parameter total: total number of images in the plot
@:parameter row: number of rows in the plot
@:parameter col: number of columns in the plot
'''
def plot_filtered_images(filters, image, n, total, row, col):
    with torch.no_grad():
        items = []
        for i in range(n):
            items.append(filters[i])
            filtered_image = cv2.filter2D(np.array(image), ddepth=-1, kernel=np.array(filters[i]))
            items.append(filtered_image)
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            plt.imshow(items[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()



if __name__ == "__main__":
    main(sys.argv)