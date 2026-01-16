import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model1
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy

workspaces_path= os.getenv('PYTHONPATH')
workspaces_path = "/workspaces/Trainers/"
print(f"Current Path: {workspaces_path}")

model_name = 'MNIST/FC/model1'
names = {'project':'MNIST'}



    
def get_loaders(batch_size=1024):
    SEED = 5700
    torch.manual_seed(SEED)
    
    path = os.path.join(workspaces_path, 'MNIST', 'data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Flatten here if your model1.Net expects (Batch, 784)
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])

    # 1. Load Standard Datasets
    train_set = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=True)
    
    # 2. Split for Validation (80/20)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])
    
    # 3. Create Loaders
    # num_workers=0 is often faster for small datasets like MNIST on Windows/Laptops
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader, test_loader
    
     
        
#GET NET
def get_untrained_net():
    net = model1.Net(28*28, num_classes=10)
    return net

def train_net(force_train=False, run_id="1", fn=None, kwargs={}): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir = os.path.join(workspaces_path,'MNIST', 'FC','model1', 'nn_models/')
    path_exists = os.path.exists(model_dir + 'best_model.pth')
    
    names['run_id']= run_id
    names['name']= f"{model_name}/{run_id}"

    if path_exists:
        checkpoint = torch.load(model_dir + 'best_model.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        checkpoint = torch.load(model_dir + 'trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully.")  
        
    if not path_exists or force_train:   
        t.train_network(trainloader, valloader, testloader,
                        root_path= model_dir, 
                        optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                        lfn=  nn.CrossEntropyLoss(), 
                        num_epochs = 10,
                        names=names, net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)  
       
        
    return trainloader, valloader, testloader, init_net, net
    

def main():
    train_net(force_train=True)

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()