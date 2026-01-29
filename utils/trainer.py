
import torch
import wandb
import time
from torch.amp import autocast
from utils.special_logger import CNNLogger, BaseLogger
scaler = torch.amp.GradScaler('cuda')
fn_data = {}

'''
Optimization: All the specialized logs like log_cnn_visuals
can be merged into val_step. This affects readability but improves speed.
''' 


def train_network(train_loader, val_loader, test_loader, net, optimizer, lfn,
                  scheduler=None, names=None, num_epochs = 5):

    # Allow PyTorch to use the TensorFloat32 (TF32) tensor cores on Ampere and newer GPUs
    # for better performance with float32 matrix multiplications.
    torch.set_float32_matmul_precision('high')

    net.cuda()
    # Optimization: Use Channels Last memory format for faster CNNs on Tensor Cores
    net = net.to(memory_format=torch.channels_last)

    # This is supposed to improve performance but I found mixed results 
    # with diminisghing returns
    #net = torch.compile(net)

    print("Initializing Wandb:")
    logger = BaseLogger(names, optimizer, lfn, net, scheduler, val_loader)

    
    wandb.watch(net, log="all", log_freq=100, idx=0)
    best_test_acc = 0
    best_test_loss = 0

    start_epoch = logger.start_epoch
    best_val_acc = logger.best_val_acc
    best_val_loss = logger.best_val_loss
    best_state_dict = logger.best_state_dict
    inputs = logger.inputs

    for i in range(start_epoch, start_epoch + num_epochs):

        print("EPOCH: ", i)
        #Train loss and accuracy are calculated on the fly during backprob for each epoch
        train_loss, train_acc = train_step(net, optimizer, lfn, train_loader)
        # Validation loss and accuracy are calculated after backprob for each epoch
        val_loss, val_acc, val_preds, val_targets = val_step(net, val_loader, lfn)       

        log_data = {
            "epoch": i,
            "train/accuracy": train_acc,
            "train/loss": train_loss,
            "val/accuracy": val_acc,
            "val/loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }      

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state_dict = net.state_dict()

            # Log wandb artifact from memory without saving to a local file      
            artifact = wandb.Artifact(f"model-{names['run_id']}", type='model', metadata={"best_val_acc": best_val_acc, "epoch": i})
            with artifact.new_file('best_model.pth', mode='wb') as f:
                torch.save({'state_dict': best_state_dict}, f)
            wandb.log_artifact(artifact)
            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_val_loss"] = best_val_loss
            
            logger.log_confusion_matrix(y_true=val_targets, preds=val_preds,
                                        epoch=i, class_names=[str(i) for i in range(10)],
                                        log_key="Validation Confusion Matrix")

            logger.log_predictions_table(net, val_loader, epoch=i, log_key="Validation Predictions", limit=256)

            #Specialized layer logs
            if inputs is not None:
                specialized_visuals_dispatcher(net, inputs, epoch=i)

        wandb.log(log_data)
        
    artifact = wandb.Artifact(f"checkpoint-{names['run_id']}", type='model', metadata={"val_acc": val_acc, "best_val_acc": best_val_acc, "epoch": i})
    with artifact.new_file('last_model.pth', mode='wb') as f:
        torch.save({
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }, f)
    wandb.log_artifact(artifact)
    wandb.run.summary["best_val_accuracy"] = best_val_acc
    wandb.run.summary["best_val_loss"] = best_val_loss
    


    # Load the model with best val accuracy before final test evaluation. 
    # Currently the weights are being taken from checkpoint
    if best_state_dict:
        net.load_state_dict(best_state_dict)

    best_test_loss, best_test_acc, test_preds, test_targets = val_step(net, test_loader, lfn)
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss
    
    logger.log_confusion_matrix(y_true=test_targets, preds=test_preds,
                                epoch=i, class_names=[str(i) for i in range(10)],
                                log_key="Test Confusion Matrix")

    logger.log_predictions_table(net, test_loader, epoch=i, log_key="Test Predictions", limit=256)
    logger.finish()
    
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
    # Accumulate on GPU to avoid CPU-GPU sync in the loop
    train_loss_accum = torch.tensor(0.0, device='cuda')
    correct_accum = torch.tensor(0.0, device='cuda')
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        # Optimization: set_to_none=True skips zeroing the memory, which is faster
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = batch
        targets = labels
        
        # Optimization: Channels Last for inputs matches the model layout
        inputs = inputs.to(device='cuda', memory_format=torch.channels_last, non_blocking=True)
        target = targets.cuda(non_blocking=True)

        with autocast(device_type='cuda'):
            output = net(inputs)
            loss = lfn(output, target)
        
        scaler.scale(loss).backward()  
        
        scaler.step(optimizer)    
        scaler.update() 

        train_loss_accum += loss.detach() * inputs.size(0)
        # Note: loss.item() triggers a CPU-GPU sync
        if batch_idx % 10 == 0:
            wandb.log({"Batch/loss": loss.item()})
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        if len(target.size()) > 1:
            _, labels_idx = torch.max(target, -1)
        else:
            labels_idx = target
        correct_accum += (predicted == labels_idx).sum()
        
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss_accum.item() / len(train_loader.dataset)
    train_acc = 100 * correct_accum.item() / total
    return train_loss, train_acc


def val_step(net, val_loader, lfn):
    global scaler
    net.eval()
    val_loss_accum = torch.tensor(0.0, device='cuda')
    correct_accum = torch.tensor(0.0, device='cuda')
    total = 0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        inputs = inputs.to(device='cuda', memory_format=torch.channels_last, non_blocking=True) 
        target = targets.cuda(non_blocking=True)
        
        with torch.no_grad():
            with autocast(device_type='cuda'): 
                output = net(inputs)
                loss = lfn(output, target)
            
            val_loss_accum += loss.detach() * inputs.size(0)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            if len(target.size()) > 1:
                _, labels_idx = torch.max(target, -1)
            else:
                labels_idx = target
            
            correct_accum += (predicted == labels_idx).sum()
            all_preds.append(predicted)
            all_targets.append(labels_idx)
        
    val_loss = val_loss_accum.item() / len(val_loader.dataset)
    val_acc = 100 * correct_accum.item() / total
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_targets = torch.cat(all_targets).cpu().tolist()
    
    return val_loss, val_acc, all_preds, all_targets



def specialized_visuals_dispatcher(net, inputs, epoch):
    """
    Dispatches to specialized visual loggers based on model type.
    This function uses a generic hook-based mechanism to capture and log layer details.
    """
    layer_handlers = {}
    layer_handlers[torch.nn.Conv2d] = CNNLogger(100, 100)

    if not layer_handlers:
        return

    net.eval()
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in net.named_modules():
        if type(layer) in layer_handlers:
            hooks.append(layer.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        net(inputs)

    for name, layer in net.named_modules():
        if type(layer) in layer_handlers:
            handler = layer_handlers[type(layer)]
            activation = activations.get(name)
            handler.call(layer, name, activation, epoch)
    
    for h in hooks:
        h.remove()


'''

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

    '''
