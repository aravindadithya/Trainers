import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.amp import autocast
import io
import os


class BaseLogger:

    def __init__(self, names, optimizer, lfn, net, scheduler=None, val_loader=None):
        
        self.names = names
        self.optimizer = optimizer
        self.lfn = lfn
        self.net = net
        self.scheduler = scheduler

        self._initialize_wandb(names, optimizer, lfn, net, scheduler)

        self.start_epoch = 1
        self.best_val_acc = 0
        self.best_val_loss = float("inf")
        self.best_state_dict = None
        
        self.inputs = self._get_viz_inputs(val_loader)

        if wandb.run.resumed:
            self._resume_run()
        else:
            self._log_initial_artifacts(self.inputs)

    def _initialize_wandb(self, names, optimizer, lfn, net, scheduler=None):
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
            "model_architecture": type(net).__name__,
            "model_structure": str(net),
            "num_parameters": sum(p.numel() for p in net.parameters()),
            "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
            "scheduler": type(scheduler).__name__ if scheduler else "None",
        }

        wandb.login(key=os.getenv('WANDB_API_KEY'))

        wandb.init(project=project, name=run_name, resume="allow", id=id, config=config, entity="Trainers100")

        # Define 'epoch' as the step metric for all epoch-level logs
        wandb.define_metric("epoch")
        metrics_to_sync = [
            "train/accuracy", "train/loss", "val/accuracy", "val/loss",
            "learning_rate", "Validation Confusion Matrix", "Validation Predictions",
            "Test Confusion Matrix",
            "Test Predictions", "fixed_val_images"
        ]
        for metric in metrics_to_sync:
            wandb.define_metric(metric, step_metric="epoch")

        for name, layer in net.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                wandb.define_metric(f"Weights_Images/{name}", step_metric="epoch")
                wandb.define_metric(f"Activation_Images/{name}", step_metric="epoch")

    def _get_viz_inputs(self, val_loader):
        inputs = None
        if val_loader is None:
            return None
        try:
            val_iter = iter(val_loader)
            inputs, _ = next(val_iter)
            inputs = inputs[:20]
            inputs = inputs.cuda(non_blocking=True)
        except StopIteration:
            print("Viz setup failed: val_loader appears to be empty. Cannot fetch a batch for visualization.")
        except Exception as e:
            print(f"Viz setup failed with an unexpected error: {e}")
        return inputs

    def _resume_run(self):
        print("Resuming from previous run...")
        self.best_val_acc = wandb.run.summary.get("best_val_accuracy", 0)
        self.best_val_loss = wandb.run.summary.get("best_val_loss", float("inf"))
        try:
            # Load checkpoint to resume training state
            artifact = wandb.use_artifact(f"checkpoint-{self.names['run_id']}:latest")
            self.start_epoch = artifact.metadata.get("epoch", 0) + 1
            path = artifact.get_entry('last_model.pth').download()
            checkpoint = torch.load(path, weights_only=True)
            self.net.load_state_dict(checkpoint['state_dict'])

            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load the state dict from the model with the best validation accuracy
            try:
                best_model_artifact = wandb.use_artifact(f"model-{self.names['run_id']}:latest")
                best_model_path = best_model_artifact.get_entry('best_model.pth').download()
                best_model_checkpoint = torch.load(best_model_path, weights_only=True)
                self.best_state_dict = best_model_checkpoint['state_dict']
            except Exception:
                # This is expected if no "best model" has been saved yet.
                pass

            print(f"Best Val Acc so far in the training run: {self.best_val_acc}")
        except Exception as e:
            print(f"Failed to resume from checkpoint artifact: {e}")

    def _log_initial_artifacts(self, inputs):
        if inputs is not None:
            wandb.log({"fixed_val_images": [wandb.Image(vutils.make_grid(inputs[i].unsqueeze(0), normalize=True), caption=f"Img {i}") for i in range(len(inputs))], "epoch": 0})

        # Log initial weights for new runs
        artifact = wandb.Artifact(f"init-weights-{self.names['run_id']}", type='model', metadata={"epoch": 0})
        with artifact.new_file('init_model.pth', mode='wb') as f:
            torch.save({'state_dict': self.net.state_dict()}, f)
        wandb.log_artifact(artifact)
        
        if inputs is not None:
            self.net.eval()
            dummy_input = inputs[0].unsqueeze(0).cuda() 
            artifact = wandb.Artifact(f"onnx-{self.names['run_id']}", type='model')
            onnx_buffer = io.BytesIO()
            torch.onnx.export(self.net, dummy_input, onnx_buffer, input_names=['input'], output_names=['output'])
            with artifact.new_file('model.onnx', mode='wb') as f:
                f.write(onnx_buffer.getvalue())
            wandb.log_artifact(artifact)
            self.net.train()

    def log_confusion_matrix(self, y_true, preds, epoch, class_names, log_key):
        wandb.log({
            log_key: wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=preds,
                class_names=class_names),
            "epoch": epoch
        })

    def log_predictions_table(self, net, loader, epoch, log_key, limit=None):
        net.eval()
        #TODO: Handle cases where input is not an image data
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

                done = False
                for j in range(len(inputs)):
                    if limit is not None and count >= limit:
                        done = True
                        break
                    
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
                if done:
                    break

        if count > 0:
            wandb.log({log_key: table, "epoch": epoch})

    def finish(self):
        wandb.finish()


class CNNLogger:

    def __init__(self, image_limits=100, weight_limits=100):
        self.image_limits = image_limits
        self.weight_limits = weight_limits

    def log_conv2d_visuals(self, layer, layer_name, activation, epoch):
        """Handler for logging Conv2d weights and activations."""
        visuals = {'epoch': epoch}
        # Log weights
        w = layer.weight.data
        op, ip, q, s = w.shape
        w = w.reshape(op * ip, 1, q, s)
        if w.shape[0] > self.weight_limits:
            w = w[:self.weight_limits]
        grid_w = vutils.make_grid(w, nrow=ip  , normalize=True, scale_each=False)
        visuals[f"Weights_Images/{layer_name}"] = wandb.Image(grid_w, caption=f"Weights_Images {layer_name}")

        # Log activations
        if activation is not None:
            act = activation
            n, ip_act, q_act, s_act = act.shape
            act = act.reshape(n * ip_act, 1, q_act, s_act)
            grid_a = vutils.make_grid(act, nrow=ip_act, normalize=True, scale_each=False)
            visuals[f"Activation_Images/{layer_name}"] = wandb.Image(grid_a, caption=f"Activations_Images {layer_name}")
        
        wandb.log(visuals)

    def call(self, layer, layer_name, activation, epoch):
        self.log_conv2d_visuals(layer, layer_name, activation, epoch)           
        