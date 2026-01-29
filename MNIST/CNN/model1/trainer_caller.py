import os
import torch
import torchvision
import torchvision.transforms as transforms
import model1
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

model_name = 'MNIST/CNN/model1'
names = {'project':'MNIST', 'type':'CNN'}


# ACCESS LOADERS
def get_loaders():
    
    SEED = 5700
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation for MNIST
        ])

    path= workspaces_path + '/MNIST/data' 
    trainset = torchvision.datasets.MNIST(root= path, train=True, download=True, transform=transform)
    train_len = int(len(trainset) * 0.8)
    val_len = len(trainset) - train_len
    trainset, valset = torch.utils.data.random_split(trainset, [train_len, val_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                                shuffle=False, num_workers=10, pin_memory=True)
    
    testset = torchvision.datasets.MNIST(root= path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10, pin_memory=True)

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model1.ConvNet()
    return net

def train_net(force_train=False, run_id="1", fn=None, kwargs={}):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders()
    model_dir= os.path.join(workspaces_path , 'MNIST', 'CNN', 'model1', 'nn_models/')   
    path_exists = os.path.exists(model_dir + 'best_model.pth')
    
    names['run_id']= run_id
    names['name']= f"{model_name}"

    if path_exists:
        checkpoint = torch.load(model_dir+'trained_nn_0.pth', weights_only=True)
        init_net.load_state_dict(checkpoint['state_dict'])  # Access the 'state_dict' within the loaded dictionary
        checkpoint = torch.load(model_dir+'best_model.pth', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully.")   
        
    if not path_exists or force_train:  
        t.train_network(trainloader, valloader, testloader,
                    root_path= model_dir, 
                    optimizer=torch.optim.SGD(net.parameters(), lr=.1),
                    lfn=  nn.CrossEntropyLoss(), 
                    num_epochs = 20,
                    names=names, net=net, init_net= init_net, save_init= not force_train, fn=fn, kwargs=kwargs)
           
        
    return trainloader, valloader, testloader, init_net, net 
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()