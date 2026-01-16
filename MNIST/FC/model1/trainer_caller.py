import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model1
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
import trainer as t
from copy import deepcopy

workspaces_path= os.getenv('PYTHONPATH')
workspaces_path= "dummy"
print(f"Current Path: {workspaces_path}")

model_name = 'MNIST/FC/model1'
names = {'project':'Training'}

def get_indices_by_class(dataset, max_per_class=500):
    """
    Returns a list of indices, picking up to 'max_per_class' 
    for each of the 10 MNIST digits.
    """
    all_indices = []
    targets = dataset.targets
    
    for digit in range(10):
        # Find all indices where the label matches this digit
        digit_indices = (targets == digit).nonzero(as_tuple=True)[0]
        
        # Pick only what is available (up to max_per_class)
        count = min(len(digit_indices), max_per_class)
        all_indices.extend(digit_indices[:count].tolist())
        
    return all_indices


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, indices):
        self.dataset = full_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map subset index to original dataset index
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        
        # Flatten (3, 32, 32) -> (3072,)
        img_flat = img.view(-1) 
        
        # One-hot encode label efficiently
        one_hot = torch.zeros(10)
        one_hot[label] = 1.0
        
        return img_flat, one_hot
    
# ACCESS LOADERS
def get_loaders(n_train_per_class=50000, n_test_per_class=10000):
    SEED = 5700
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    #cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    path= workspaces_path + '/trained_models/MNIST/data'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load raw datasets (doesn't load images into RAM yet)
    raw_train = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
    raw_test = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=True)
    
    # Get Balanced Indices
    train_indices = get_indices_by_class(raw_train, n_train_per_class)
    test_indices = get_indices_by_class(raw_test, n_test_per_class)
    
    # Split Train into Train/Val (80/20)
    train_idx, val_idx = train_test_split(train_indices, train_size=0.8, random_state=SEED)
    
    train_loader = DataLoader(MNISTDataset(raw_train, train_idx), batch_size=100, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(MNISTDataset(raw_train, val_idx), batch_size=100, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(MNISTDataset(raw_test, test_indices), batch_size=128, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
    
     
        
#GET NET
def get_untrained_net():
    net = model1.Net(28*28, num_classes=10)
    return net

def train_net(force_train=False, run_id=1, fn=None, kwargs={}): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir = os.path.join(workspaces_path,'trained_models', 'MNIST', 'model1', 'nn_models/')
    path_exists = os.path.exists(model_dir + f'{names['name']}_trained_nn.pth')
    
    names['run_id']= {run_id}
    names['name']= f"{model_name}/{run_id}"

    if path_exists:
        checkpoint = torch.load(model_dir + f'{names['name']}_trained_nn.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        checkpoint = torch.load(model_dir + f'{names['name']}_trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully.")  
        
    if not path_exists or force_train:   
        t.train_network(trainloader, valloader, testloader,
                        num_classes=10, root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                        lfn=  nn.MSELoss(), 
                        num_epochs = 10,
                        names=names, net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)  
       
        
    return trainloader, valloader, testloader, init_net, net
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()