import os
import torch
import torchvision
import torchvision.transforms as transforms
import model1
import torch.nn as nn
from utils import trainer as t
import wandb
from utils.special_schedulers import CosineAnnealingWarmRestartsDecay

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                                shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)
    
    testset = torchvision.datasets.MNIST(root= path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True)

    return trainloader, valloader, testloader

#GET NET
def get_untrained_net():
    net= model1.ConvNet()
    #Channels last is faster for ConvNets on GPU
    net = net.to(memory_format=torch.channels_last)
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
    lfn = nn.CrossEntropyLoss()

    config = {
        "project": names['project'],
        "entity": "Trainers100",
        "run_name": names['name'],
        "run_id": run_id,
        "net": net,
        "train_loader": trainloader,
        "val_loader": valloader,
        "test_loader": testloader,
        "optimizer": optimizer,
        "lfn": lfn,
        "scheduler": scheduler
    }

    t.train_network(config, num_epochs=epochs)
    

def main():
    train_net()

if __name__ == "__main__":
    #For some reason executing through console adds 4sec delay
    main()