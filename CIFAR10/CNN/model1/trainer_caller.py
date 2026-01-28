import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast
import torch.nn as nn
from utils import trainer as t
from copy import deepcopy
import wandb

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

model_name = 'CIFAR10/CNN/model1'
names = {'project':'CIFAR10', 'type':'CNN'}


# ACCESS LOADERS
def get_loaders():

    SEED = 5700
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                                shuffle=False, num_workers=10, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10, pin_memory=True)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, valloader, testloader

def get_acc_ce(net, loader):
    global scaler
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)  # Move to CUDA consistently
            with autocast(device_type='cuda'):
               outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total += targets.size(0)
            # Targets maybe in one-hot format. Hence Max
            if len(targets.size()) > 1:
                _, labels = torch.max(targets, -1)
            else:
                labels = targets
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#GET NET
def get_untrained_net():
    net = torchvision.models.resnet18(weights=None)
    # Modify the model for CIFAR-10.
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(net.fc.in_features, 10)
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

def train_net(run_id="1", epochs=10, fn=None, kwargs={}):

    net = get_untrained_net()
    init_net = deepcopy(net)
    trainloader, valloader, testloader = get_loaders() 
    names['run_id']= run_id
    names['name']= f"{model_name}"

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t.train_network(trainloader, valloader, testloader,
                    optimizer=optimizer,
                    lfn=  nn.CrossEntropyLoss(), 
                    num_epochs = epochs,
                    names=names, net=net, init_net= init_net, scheduler=scheduler, fn=fn, kwargs=kwargs)
           
    
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()