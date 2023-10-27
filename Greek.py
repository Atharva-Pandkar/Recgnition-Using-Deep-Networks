# import statements
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# greek data set transform
import Helper
import Networks


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
'''
Load the given Greek dataset, which contains examples of alpha, beta, and gamma(3 for each)
Write the intensity values and category of the images to two csv files
Load a digit embedding model, which inherits MyNetwork and terminates at the Dense layer with 50 outputs
Load the MNIST training dataset, apply the model to the first image and print out the shape of the result
Apply the model to the Greek dataset and get the element vectors
Compute the ssd of some example images and plot the result
Load a custom Greek digit dataset
Apply the model to the custom Greek dataset and get the element vectors
Compute the ssd of the custom images and plot the result
'''
def main(argv):
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder("greek",
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=1,
        shuffle=False)
    # Task 3
    # load greek dataset
    model = Networks.MNIST_Network()
    model.load_state_dict(torch.load('results\\model_state_dict.pth'))
    model.fc2 = torch.nn.Linear(50,3)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 1
    train_losses = []
    train_counter = []
    for epoch in range(num_epochs):
        for i, data in enumerate(greek_train, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % Networks.LOG_INTERVAL == 0:
                train_losses.append(loss.item())
                train_counter.append(
                    (i * 64) + ((epoch - 1) * len(greek_train.dataset)))
                torch.save(model.state_dict(), 'results\\model.pth')
                torch.save(optimizer.state_dict(), 'results\\optimizer.pth')



    Helper.test_custom_greek(model,greek_train,optimizer,0)


    # load custom greek symbols
    custom_image_dir = 'C:\\Users\\athar\\PycharmProjects\\pythonProject\\Assingment5\\custom_greek'
    custom_greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(custom_image_dir,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=9,
        shuffle=False)
    los =[]
    for x in range(0,100):
        Helper.test_custom_greek(model=model,greek_custom=custom_greek_train,optimizer=optimizer,epochs=x)

    return


if __name__ == "__main__":
    main(sys.argv)


