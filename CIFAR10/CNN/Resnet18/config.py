import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import model

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

# ACCESS LOADERS
def get_loaders():

    SEED = 7500
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    means = (0.4914, 0.4822, 0.4465)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])

    path= workspaces_path + '/CIFAR10/data' 
    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_len = int(len(trainset) * 0.8)
    val_len = len(trainset) - train_len
    trainset, valset = torch.utils.data.random_split(trainset, [train_len, val_len])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                                shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)
    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, valloader, testloader



#GET NET
def get_untrained_net():
    net = model.ResNet18()
    #Channels last is faster for ConvNets on GPU
    net = net.to(memory_format=torch.channels_last)
    return net


def get_config(run_id="1", project="CIFAR10", entity="Trainers100", run_name="Resnet18_CNN"):

    net = get_untrained_net()
     
    trainloader, valloader, testloader = get_loaders() 

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lfn = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
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
        "net": net,
        "train_loader": trainloader,
        "val_loader": valloader,
        "test_loader": testloader,
        "optimizer": optimizer,
        "lfn": lfn,
        "scheduler": scheduler
    }

    return config
        