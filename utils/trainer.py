
import torch
import time
import os
from torch.amp import autocast
scaler = torch.amp.GradScaler('cuda')
fn_data = {}
import wandb


def initialize_wandb(names, optimizer, lfn, net, num_epochs):
    
    project = names['project']
    run_name = names['name']
    id = names['run_id']

    # Extracting config details
    config = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer": type(optimizer).__name__,
        "loss_function": type(lfn).__name__,
        "epochs": num_epochs,
        "model_architecture": type(net).__name__,
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
    }

    #TODO: REMOVE THIS KEY BEFORE PUSHING
    wandb.login(key ='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')

    wandb.init(project=project, name=run_name, resume=True, id=id, config=config)



def train_network(train_loader, val_loader, test_loader, net, init_net, optimizer, lfn, root_path,
                  names=None, num_epochs = 5, save_init= False, fn=None, kwargs={}):


    params = 0
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    print("Initializing Wandb:")
    initialize_wandb(names, optimizer, lfn, net, num_epochs)

    net.cuda()
    wandb.watch(net, log="all", log_freq=100)
    #net.to(dtype=torch.float32, device='cuda')
    best_val_acc = 0
    best_test_acc = 0
    #best_val_loss = np.float("inf")
    best_val_loss = float("inf")
    best_test_loss = 0
    os.makedirs(root_path, exist_ok=True)
    for i in range(num_epochs):

        print("EPOCH: ", i)

        if save_init and (i == 0 or i == 1):
            #net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()    
            file_path = os.path.join(root_path, f'trained_nn_{i}.pth')
            torch.save(d, file_path)
            #net.cuda()
        if fn is not None:
            #net.to(dtype=torch.float32, device='cuda')
            #init_net.to(dtype=torch.float32, device='cuda')
            kwargs = {'net': net, 'train_loader': train_loader, 'init_net': init_net, **kwargs}
            fn_data[i]= fn(epoch=i, kwargs=kwargs)
            net.to(dtype=torch.float32, device='cuda')
            init_net.to(dtype=torch.float32, device='cpu')

        train_loss = train_step(net, optimizer, lfn, train_loader)
        val_loss = val_step(net, val_loader, lfn)       
        train_acc = get_acc_ce(net, train_loader)
        val_acc = get_acc_ce(net, val_loader)           

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            #net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            file_path = os.path.join(root_path, 'best_model.pth')
            torch.save(d, file_path)
            #net.cuda()
        '''
        if val_loss <= best_val_loss:
            #best_val_loss = val_loss
            #best_test_loss = test_loss
            '''
        '''
        wandb.log({
            "train/accuracy": train_acc,
            "train/loss": train_loss,
            "val/accuracy": val_acc,
            "val/loss": val_loss,
        })'''

    checkpoint = torch.load(os.path.join(root_path, 'best_model.pth'), weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])

    best_test_loss = val_step(net, test_loader, lfn)
    best_test_acc = get_acc_ce(net, test_loader)
    
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss
    wandb.run.summary["best_val_accuracy"] = best_val_acc
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.finish()
    
    print("FINISHED TRAINING :)")

    '''

    if fn and fn_data:
        torch.save(fn_data, kwargs['save_path'])
        '''

def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train_step(net, optimizer, lfn, train_loader):
    global scaler
    net.train()
    start = time.time()
    train_loss = 0.

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        
        inputs = inputs.cuda(non_blocking=True)
        target = targets.cuda(non_blocking=True)

        with autocast(device_type='cuda'):
            output = net(inputs)
            loss = lfn(output, target)
        
        scaler.scale(loss).backward()  
        
        scaler.step(optimizer)    
        scaler.update() 

        wandb.log({"batch/loss": loss.item()})               
        train_loss += loss.detach().item() * len(inputs)
        
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader, lfn):
    global scaler
    net.eval()
    val_loss = 0.

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        inputs = inputs.cuda(non_blocking=True) 
        target = targets.cuda(non_blocking=True)
        
        with torch.no_grad():
            with autocast(device_type='cuda'): 
                output = net(inputs)
                loss = lfn(output, target)
                
            val_loss += loss.detach().item() * len(inputs)
        
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


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

