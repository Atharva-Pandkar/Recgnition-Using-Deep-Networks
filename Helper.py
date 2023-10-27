import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import Networks


def plot_images(data, row, col):
    images = enumerate(data)
    batch_idx, (image_data, target_image) = next(images)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(image_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(target_image[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train(epoch, model, optimizer, train_loader, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % Networks.LOG_INTERVAL == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


'''
The function is the testing model and print the accuracy information
@:parameter model: the model to be tested
@:parameter test_loader: the test data
@:parameter test_losses: array to record test losses
'''


def test(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


'''
The function plots curves of the training loses and testing losses
@:parameter train_counter: array of train counter
@:parameter train_losses: array of train losses
@:parameter test_counter: array of test counter
@:parameter test_losses: array of test losses
'''


def plot_curve(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


'''
The function apply model on dataset and get the first n data and the labels
@:parameter data: the testing data
@:parameter model: the model used
@:return first_n_data: array contains the first n data
@:return first_n_label: array contains the label of the first n data
'''
def first_n_output(data, model,n):
    first_n_data = []
    first_n_label = []

    count = 0
    for data, target in data:
        if count < n:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_n_data.append(squeeze_data)

            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - output: {output}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - prediction label: {label}')
                first_n_label.append(label)
                count += 1
        else:
            return first_n_data , first_n_label
    return first_n_data, first_n_label


def test_custom_greek(model,greek_custom,optimizer,epochs) :
    for params in model.parameters():
        params.requires_grad = False
    batch_accuracy =0
    totaa =0
    correct=0
    with torch.no_grad():
        mp=[]
        for images, labels in greek_custom:


            # Pass the batch through the network and get the predicted classes
            output = model(images)
            a, predicted = torch.max(output,1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            mp.append(accuracy)
            batch_accuracy +=correct
            totaa+= total
            # Print the accuracy for this batch
            #print("Batch accuracy: {:.2f}%".format(accuracy * 100))
        print("Total Acuuracy : {:.2f}%".format((sum(mp)/len(mp))*100)+" Epoch number",epochs)
