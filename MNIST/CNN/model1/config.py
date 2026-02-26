import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import model1
from utils.special_schedulers import CosineAnnealingWarmRestartsDecay

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

def get_loaders():
    SEED = 5700
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    path= workspaces_path + '/MNIST/data' 
    trainset = torchvision.datasets.MNIST(root= path, train=True, download=True, transform=transform)
    train_len = int(len(trainset) * 0.8)
    val_len = len(trainset) - train_len
    trainset, valset = torch.utils.data.random_split(trainset, [train_len, val_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                                shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)
    
    testset = torchvision.datasets.MNIST(root= path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)

    return trainloader, valloader, testloader

def get_untrained_net():
    net= model1.ConvNet()
    #Channels last is faster for ConvNets on GPU
    net = net.to(memory_format=torch.channels_last)
    return net

def get_config(run_id="1", project="MNIST", entity="Trainers100", run_name="CNN_Model1"):
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