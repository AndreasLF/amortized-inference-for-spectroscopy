import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import dill
import os
import sys

# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

def flatten(x):
    """Transforms a 28x28 image into a 784 vector
    From 02456 Deep Learning course https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch
    
    
    Args:
        x (Tensor): a 28x28 image

    Returns:
        Tensor: a 784 vector
    """
    tt = torchvision.transforms.ToTensor()
    return tt(x).view(28**2)

def stratified_sampler(labels, classes):
    """Sampler only picks datapoints in specified classes
    From 02456 Deep Learning course https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch
    
    Args:
        labels (Tensor): a tensor of labels
        classes (list): a list of classes to sample from
    
    Returns:
        SubsetRandomSampler: a sampler that only picks datapoints in specified classes
    
    """
    (indices,) = np.where(np.array([i in classes for i in labels]))
    indices = torch.from_numpy(indices)
    return torch.utils.data.sampler.SubsetRandomSampler(indices)

def plot_loss(epochs, train_losses, test_losses):
    """Plot the training and validation loss
    
    Args:
        epochs (int): The number of epochs
        train_losses (list): The training loss
        test_losses (list): The validation loss
    """
    
    plt.title("Error")
    # Set the x axis label of the current axis.
    plt.xlabel("Epoch")
    # Set the y axis label of the current axis.
    plt.ylabel("Error")

    plt.plot(np.arange(epochs), train_losses, color="black")
    plt.plot(np.arange(epochs), test_losses, color="gray", linestyle="--")
    plt.legend(['Training error', 'Validation error'])

    plt.show()

# Load model file
dill_file = "MNIST_autoencoder_2023-03-14_5epochs.dill"
with open(f'{results_dir}\\{dill_file}', 'rb') as f:
    model_data = dill.load(f)

# Unpack model data
autoencoder = model_data['model']
datee = model_data['date']
epochs = model_data['epochs']
train_losses = model_data['train_loss']
test_losses = model_data['test_loss']

# Plot the training and validation loss
plot_loss(epochs, train_losses, test_losses)


# Load the MNIST data
dset_train = torchvision.datasets.MNIST(f"{data_dir}/MNIST/", train=True, transform=flatten)
dset_test  = torchvision.datasets.MNIST(f"{data_dir}/MNIST/", train=False, transform=flatten)

classes = [0,1]
batch_size = 64
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.targets, classes), pin_memory=cuda)
test_loader  = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.targets, classes), pin_memory=cuda)


# Loop through points in dataset and plot them
for i, (x, y) in enumerate(train_loader):
    z = autoencoder.encoder(x.to(device))
    z = z.to('cpu').detach().numpy()

    print(z.shape)
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    break
plt.show()