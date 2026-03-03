import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import model1
from utils.special_schedulers import CosineAnnealingWarmRestartsDecay

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

def get_loaders(batch_size=1024):
    SEED = 5700
    torch.manual_seed(SEED)
    
    path = workspaces_path + '/MNIST/data'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])

    train_set = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=True)
    
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader

def get_untrained_net():
    net = model1.Net(28*28, num_classes=10)
    return net

def get_config(run_id="1", project="MNIST", entity="Trainers100", run_name="FC_Model1"):
    net = get_untrained_net()
    trainloader, valloader, testloader = get_loaders()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    epochs = 10
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=int(epochs/3)+1, decay=0.8)
    lfn = nn.CrossEntropyLoss()

    config = {
        "project": project,
        "entity": entity,
        "run_name": run_name,
        "run_id": run_id,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer_name": type(optimizer).__name__,
        "loss_function_name": type(lfn).__name__,
        "model_architecture": type(net).__name__,
        "model_structure": str(net),
        "num_parameters": sum(p.numel() for p in net.parameters()),
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
        "scheduler_name": type(scheduler).__name__ if scheduler else "None",
        "num_classes": 10,
        "max_images": 32,
        "rotate_inputs": False,
        "net": net,
        "train_loader": trainloader,
        "val_loader": valloader,
        "test_loader": testloader,
        "optimizer": optimizer,
        "lfn": lfn,
        "scheduler": scheduler
    }
    return config