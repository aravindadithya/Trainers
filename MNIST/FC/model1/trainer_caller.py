import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model1
import torch.nn as nn
from utils import trainer as t
import wandb
from utils.special_schedulers import CosineAnnealingWarmRestartsDecay

workspaces_path= os.getenv('PYTHONPATH')
workspaces_path = "/workspaces/Trainers/"
print(f"Current Path: {workspaces_path}")

model_name = 'MNIST/FC/model1'
names = {'project':'MNIST', 'type':'FC'}



    
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
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader
    
     
        
#GET NET
def get_untrained_net():
    net = model1.Net(28*28, num_classes=10)
    return net

def get_trained_net(run_id="1"):
    net = get_untrained_net()
    api = wandb.Api()
    try:
        artifact = api.artifact(f"Trainers100/{names['project']}/model-{run_id}:latest")
        model_dir = artifact.download()
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'), weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights from artifact: {artifact.name}")
        
    except Exception as e:
        print(f"Error loading from WandB: {e}")
    return net

def train_net(run_id="1", epochs=10):
    net = get_untrained_net()
    trainloader, valloader, testloader = get_loaders()
    names['run_id'] = run_id
    names['name'] = f"{model_name}"

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=int(epochs/3)+1, decay=0.8)

    t.train_network(trainloader, valloader, testloader,
                        optimizer=optimizer,
                        lfn=  nn.CrossEntropyLoss(), 
                        num_epochs=epochs,
                        names=names, net=net, scheduler=scheduler)
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()