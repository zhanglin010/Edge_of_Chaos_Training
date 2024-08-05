import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np

def load_data(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    datasets = {
        'mnist': torchvision.datasets.MNIST,
        'kmnist': torchvision.datasets.KMNIST,
        'emnist/mnist': lambda root, train, download, transform: torchvision.datasets.EMNIST(root=root, split='mnist', train=train, download=download, transform=transform),
        'fashion_mnist': torchvision.datasets.FashionMNIST,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} is not supported")
    
    DatasetClass = datasets[dataset_name]
    trainset = DatasetClass(root='./data', train=True, download=True, transform=transform)
    testset = DatasetClass(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    x_train, y_train = next(iter(trainloader))
    x_test, y_test = next(iter(testloader))

    x_train = x_train.view(x_train.size(0), -1).numpy()
    x_test = x_test.view(x_test.size(0), -1).numpy()
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    return (x_train, y_train), (x_test, y_test)

def load_qmnist_data():
    with open('./data/qmnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    x_train, y_train = mnist['train_data'], mnist['train_labels']
    x_test, y_test = mnist['test_data'], mnist['test_labels']

    x_train = np.reshape(x_train, [x_train.shape[0], -1]).astype('float32') / 255.
    x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype('float32') / 255.

    return (x_train, y_train), (x_test, y_test)
