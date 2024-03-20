from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import torch
from tqdm import tqdm

def train_test_val(batch_size=16):
    dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader

def save_checkpoint(state, filename="chekpoint.pt.tar"):
    torch.save(state,filename)

def load_checkpoint(model,optimizer,checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def train(model,loss,optimizer,train_loader,val_loader,train_len,val_len,epoch=5,save=False):
    for i in tqdm(range(epoch)):
        model.train()
        train_correct=0
        l=0
        for x,y in train_loader:
            #for densenets
            # x_pred=model(x.reshape(x.shape[0],-1))
            #for convents
            x_pred=model(x)
            x_pred=torch.squeeze(x_pred)
            optimizer.zero_grad()
            l=loss(x_pred,y)
            l.backward()
            optimizer.step()
            pred=x_pred.argmax(axis=1)
            train_correct+=(pred==y).sum()

        model.eval()
        val_correct=0
        with torch.no_grad():
            for x,y in val_loader:
                # for densenets
                # x_pred=model(x.reshape(x.shape[0],-1))
                # for convents
                x_pred = model(x)
                x_pred = torch.squeeze(x_pred)
                pred = x_pred.argmax(axis=1)
                val_correct += (pred == y).sum()
        print(f'\n {i} epoch, loss={l}, '
              f'\n train_accuracy={train_correct/train_len}'
              f'\n val_accuracy={val_correct/val_len}')
    if save==True:
        checkpoint={'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)