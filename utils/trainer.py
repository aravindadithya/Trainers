
import torch
import torch.nn.functional as F
import torch.onnx
import torchvision.utils as vutils
import wandb
import time
import io
from torch.amp import autocast
scaler = torch.amp.GradScaler('cuda')
fn_data = {}

'''
Optimization: All the specialized logs like log_cnn_visuals
can be merged into val_step. This affects readability but improves speed.
''' 

def initialize_wandb(names, optimizer, lfn, net, scheduler=None):
    
    project = names['project']
    run_name = names['name']
    id = names['run_id']

    # Extracting config details
    config = { 
        "project": project,
        "run_name": run_name,
        "run_id": id,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer": type(optimizer).__name__,
        "loss_function": type(lfn).__name__,
        #"epochs": num_epochs,
        "model_architecture": type(net).__name__,
        "model_structure": str(net),
        "num_parameters": sum(p.numel() for p in net.parameters()),
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
        "scheduler": type(scheduler).__name__ if scheduler else "None",
    }

    #TODO: REMOVE THIS KEY BEFORE PUSHING
    wandb.login(key ='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')

    wandb.init(project=project, name=run_name, resume=True, id=id, config=config)



def train_network(train_loader, val_loader, test_loader, net, init_net, optimizer, lfn,
                  scheduler=None, names=None, num_epochs = 5, fn=None, kwargs={}):


    
    print("Initializing Wandb:")
    initialize_wandb(names, optimizer, lfn, net, scheduler)

    net.cuda()
    wandb.watch(net, log="all", log_freq=10, idx=0)
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float("inf")
    best_test_loss = 0
    best_state_dict = None
    viz_images = None

    try:
        viz_iter = iter(val_loader)
        viz_inputs, _ = next(viz_iter)
        viz_images = viz_inputs[:20]
        wandb.log({"fixed_val_images": [wandb.Image(vutils.make_grid(viz_images[i].unsqueeze(0), normalize=True), caption=f"Img {i}") for i in range(len(viz_images))]})
        viz_images = viz_images.cuda()
        net.eval()
        dummy_input = viz_inputs[0].unsqueeze(0).cuda()
        artifact = wandb.Artifact(f"onnx-{names['run_id']}", type='model')
        onnx_buffer = io.BytesIO()
        torch.onnx.export(net, dummy_input, onnx_buffer, input_names=['input'], output_names=['output'])
        with artifact.new_file('model.onnx', mode='wb') as f:
            f.write(onnx_buffer.getvalue())
        wandb.log_artifact(artifact)
        net.train()
    except Exception as e:
        print(f"Viz setup failed: {e}")

    start_epoch = 1
    if wandb.run.resumed:
        start_epoch = wandb.run.summary.get("epoch", 0) + 1
        best_val_acc = wandb.run.summary.get("best_val_accuracy", 0)
        best_val_loss = wandb.run.summary.get("best_val_loss", float("inf"))
        try:
            artifact = wandb.use_artifact(f"checkpoint-{names['run_id']}:latest")
            path = artifact.get_entry('last_model.pth').download()
            checkpoint = torch.load(path, weights_only=True)
            net.load_state_dict(checkpoint['state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_state_dict = checkpoint['state_dict']
            print(f"Resumed from artifact. Best Val Acc: {best_val_acc}")
        except Exception as e:
            print(f"Failed to resume from artifact: {e}")
    else:
        # Log initial weights for new runs
        artifact = wandb.Artifact(f"init-weights-{names['run_id']}", type='model', metadata={"epoch": 0})
        with artifact.new_file('init_model.pth', mode='wb') as f:
            torch.save({'state_dict': net.state_dict()}, f)
        wandb.log_artifact(artifact)

    for i in range(start_epoch, start_epoch + num_epochs):

        print("EPOCH: ", i)

        if fn is not None:
            kwargs = {'net': net, 'train_loader': train_loader, 'init_net': init_net, **kwargs}
            fn_data[i]= fn(epoch=i, kwargs=kwargs)
            net.to(dtype=torch.float32, device='cuda')
            init_net.to(dtype=torch.float32, device='cpu')

        #Train loss and accuracy are calculated on the fly during training
        train_loss, train_acc = train_step(net, optimizer, lfn, train_loader)
        # Validation loss and accuracy are calculated after each epoch
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

            # Log artifact from memory without saving to a local file
            
            artifact = wandb.Artifact(f"model-{names['run_id']}", type='model', metadata={"best_val_acc": best_val_acc, "epoch": i})
            with artifact.new_file('best_model.pth', mode='wb') as f:
                torch.save({'state_dict': best_state_dict}, f)
            wandb.log_artifact(artifact)

            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_val_loss"] = best_val_loss

            log_data["Validation Confusion Matrix"] = wandb.plot.confusion_matrix(probs=None,
                                                    y_true=val_targets, preds=val_preds,
                                                    class_names=[str(i) for i in range(10)])
            log_data["Validation Predictions"] = log_predictions_table(net, val_loader, limit=256)
        
        if names['type'] == 'CNN' and viz_images is not None:
            log_data.update(log_cnn_visuals(net, viz_images))

        wandb.log(log_data)
        
    artifact = wandb.Artifact(f"checkpoint-{names['run_id']}", type='model', metadata={"val_acc": val_acc, "epoch": i})
    with artifact.new_file('last_model.pth', mode='wb') as f:
        torch.save({
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }, f)
    wandb.log_artifact(artifact)
    

    # Load the model with best val accuracy before final test evaluation.
    if best_state_dict:
        net.load_state_dict(best_state_dict)

    best_test_loss, best_test_acc, test_preds, test_targets = val_step(net, test_loader, lfn)
    
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss
    wandb.run.summary["best_val_accuracy"] = best_val_acc
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.log({"Test Confusion Matrix": wandb.plot.confusion_matrix(probs=None,
                                                            y_true=test_targets, preds=test_preds,
                                                            class_names=[str(i) for i in range(10)])})
    wandb.log({"Test Predictions": log_predictions_table(net, test_loader, limit=256)})
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
    correct = 0
    total = 0

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
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        if len(target.size()) > 1:
            _, labels_idx = torch.max(target, -1)
        else:
            labels_idx = target
        correct += (predicted == labels_idx).sum().item()
        
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = 100 * correct / total
    return train_loss, train_acc


def val_step(net, val_loader, lfn):
    global scaler
    net.eval()
    val_loss = 0.
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

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
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            if len(target.size()) > 1:
                _, labels_idx = torch.max(target, -1)
            else:
                labels_idx = target
            correct += (predicted == labels_idx).sum().item()
            
            all_preds.extend(predicted.tolist())
            all_targets.extend(labels_idx.tolist())
        
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total
    return val_loss, val_acc, all_preds, all_targets

def log_cnn_visuals(net, images):
    net.eval()
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        net(images)

    visuals = {}
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            w = layer.weight.data
            op,ip, q, s = w.shape
            #w = w.transpose(ip, op, q, s)
            w = w.reshape(op*ip, 1, q, s)
            if w.shape[0] > 256:
                w = w[:256]
            grid_w = vutils.make_grid(w, nrow=ip, normalize=True, scale_each=False)
            visuals[f"Images/weights/{name}"] = wandb.Image(grid_w, caption=f"Weights {name}")
            
            if name in activations:
                act = activations[name]
                n, ip, q, s = act.shape
                act = act.view(n*ip, 1, q, s)
                grid_a = vutils.make_grid(act, nrow=ip, normalize=True, scale_each=False)
                visuals[f"Images/activations/{name}"] = wandb.Image(grid_a, caption=f"Activations {name}")
    
    for h in hooks:
        h.remove()
        
    return visuals


def log_predictions_table(net, loader, limit=None):
    net.eval()
    columns = ["Image", "Ground Truth", "Prediction", "Confidence"] + [f"Score_{i}" for i in range(10)]
    table = wandb.Table(columns=columns)
    
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = net(inputs)
                probs = F.softmax(outputs, dim=1)
            
            confidences, preds = torch.max(probs, 1)
            
            if len(targets.size()) > 1:
                _, targets_idx = torch.max(targets, -1)
            else:
                targets_idx = targets
                
            inputs_cpu = inputs.cpu()
            targets_cpu = targets_idx.cpu()
            preds_cpu = preds.cpu()
            confidences_cpu = confidences.cpu()
            probs_cpu = probs.cpu()
            
            for j in range(len(inputs)):
                if limit is not None and count >= limit:
                    return table
                
                if preds_cpu[j] == targets_cpu[j]:
                    continue

                img = inputs_cpu[j]
                #TODO: Handle the case when image is given as 1D tensor properly.
                if img.dim() == 1:
                    img = img.view(1, 28, 28)
                
                img_viz = vutils.make_grid(img, normalize=True)
                row = [wandb.Image(img_viz), str(targets_cpu[j].item()), str(preds_cpu[j].item()), confidences_cpu[j].item()]
                row.extend(probs_cpu[j].tolist())
                table.add_data(*row)
                count += 1
    return table


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
