import torch
from torch import nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

# MLP for tabular data
class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(MLP, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(nn.LogSoftmax(dim=1))

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x

# Basic CNN for CIFAR/SVHN data
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Basic CNN for MNIST data
class MNISTCNN(nn.Module):
    def __init__(self, n_classes=10):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# DLLP for tabular data
class BatchAvgLayer(nn.Module):
    def __init__(self):
        super(BatchAvgLayer, self).__init__()

    def forward(self, x):
        return torch.mean(input=x, dim=0)

class MLPBatchAvg(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(MLPBatchAvg, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(nn.LogSoftmax(dim=1))
        self.batch_avg = BatchAvgLayer()

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        softmax = x.clone()
        if self.training:
            x = self.batch_avg(x)
        
        return x, softmax

# DLLP for CIFAR/SVHN
class CNNBatchAvg(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNBatchAvg, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.batch_avg = BatchAvgLayer()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        softmax = x.clone()
        if self.training:
            x = self.batch_avg(x)

        return x, softmax

# DLLP for MNIST 
class MNISTBatchAvg(nn.Module):
    def __init__(self, n_classes=10):
        super(MNISTBatchAvg, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.batch_avg = BatchAvgLayer()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        softmax = x.clone()
        if self.training:
            x = self.batch_avg(x)
            
        return x, softmax
