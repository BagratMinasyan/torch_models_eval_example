import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Function import train_test_val, save_checkpoint, load_checkpoint, train
from Models import NN, CNN

model = CNN(1, 10)
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = train_test_val()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1)

# train(model,loss,optimizer,train_loader,val_loader,len(train_dataset),len(val_dataset))
model,optimizer=load_checkpoint(model,optimizer,torch.load('chekpoint.pt.tar'))


print(model)
print(optimizer)