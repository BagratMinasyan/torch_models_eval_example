import torch
import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        print(x.shape)
        return x

class CNN(nn.Module):
    def __init__(self,in_channels,output_size):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,8,kernel_size=3,)
        self.pool =nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(8,4,kernel_size=3)
        self.flat=nn.Flatten(1,-1)
        self.fc1=nn.Linear(100,output_size)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=self.flat(x)
        x=self.fc1(x)
        return x