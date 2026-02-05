import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.amp import autocast
import io
import os
import matplotlib.cm as cm
from captum.attr import LayerGradCam, LayerAttribution


class BaseLogger:

    def __init__(self, names, optimizer, lfn, net, scheduler=None, val_loader=None, rotate_inputs=True):
        
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
        self.rotate_inputs = rotate_inputs
        self.inputs, self.targets = self._get_viz_inputs(val_loader)

        if wandb.run.resumed:
            self._resume_run()
        else:
            self._log_initial_artifacts(self.inputs)

    def _initialize_wandb(self, names, optimizer, lfn, net, scheduler=None):
        
        project = names['project']
        entity = names['entity']
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

        wandb.init(project=project, name=run_name, resume="allow", id=id, config=config, entity=entity)

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
        targets = None
        if val_loader is None:
            return None, None
        try:
            val_iter = iter(val_loader)
            inputs, targets = next(val_iter)
            inputs = inputs[:20]
            targets = targets[:20]
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            if inputs.dim() == 4 and self.rotate_inputs:
                # Rotate inputs: 0, 90, 180, 270
                rotated_inputs = []
                for rot in range(4):
                    rotated_inputs.append(torch.rot90(inputs, k=rot, dims=[2, 3]))       
                # Stack: (B, 4, C, H, W)
                inputs = torch.stack(rotated_inputs, dim=1)
                inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
                targets = torch.repeat_interleave(targets, 4, dim=0)
        except StopIteration:
            print("Viz setup failed: val_loader appears to be empty. Cannot fetch a batch for visualization.")
        except Exception as e:
            print(f"Viz setup failed with an unexpected error: {e}")
        return inputs, targets

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
            try:
                # Load the state dict from the model with the best validation accuracy
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
            if inputs.dim() == 5:
                imgs = []
                for i in range(len(inputs)):
                    grid = vutils.make_grid(inputs[i], nrow=inputs.shape[1], normalize=True)
                    imgs.append(wandb.Image(grid, caption=f"Img {i}"))
                wandb.log({"fixed_val_images": imgs, "epoch": 0})
            else:
                wandb.log({"fixed_val_images": [wandb.Image(vutils.make_grid(inputs[i].unsqueeze(0), normalize=True), caption=f"Img {i}") for i in range(len(inputs))], "epoch": 0})

        # Log initial weights for new runs
        artifact = wandb.Artifact(f"init-weights-{self.names['run_id']}", type='model', metadata={"epoch": 0})
        with artifact.new_file('init_model.pth', mode='wb') as f:
            torch.save({'state_dict': self.net.state_dict()}, f)
        wandb.log_artifact(artifact)
        
        if inputs is not None:
            self.net.eval()
            dummy_input = inputs[0].unsqueeze(0).cuda() 
            if inputs.dim() == 5:
                dummy_input = inputs[0][0].unsqueeze(0).cuda()
            else:
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

    def __init__(self, inputs, targets, secondary_dim=1 , max_images=100, max_weight_filters=100):
        self.max_images = max_images
        self.max_weight_filters = max_weight_filters
        self.non_critical = os.getenv('NON_CRITICAL_LOGS', 'False').lower() in ('true', '1', 't')
        
        self.viz_inputs = None
        self.viz_targets = None
        self.secondary_dim = secondary_dim
        if inputs is not None:
            viz_batch_size = min(len(inputs), self.max_images)
            self.viz_inputs = inputs[:viz_batch_size]
            if targets is not None:
                self.viz_targets = targets[:viz_batch_size]

    def log_weights(self, layer, layer_name, epoch):
        """Handler for logging Conv2d weights."""
        w = layer.weight.data
        op, ip, q, s = w.shape
        w = w.reshape(op * ip, 1, q, s)
        if w.shape[0] > self.max_weight_filters:
            w = w[:self.max_weight_filters]
        grid_w = vutils.make_grid(w, nrow=ip, normalize=True, scale_each=False)
        wandb.log({f"Weight_Filters/{layer_name}": wandb.Image(grid_w, caption=f"Weights_Filter of {layer_name}"), "epoch": epoch})

    def log_featuremap(self, layer_name, activation, epoch):
        """Handler for logging Conv2d activations."""
        if activation is not None:
            avg_act = activation.mean(dim=1, keepdim=True)
            grid = vutils.make_grid(avg_act, nrow=self.secondary_dim, normalize=True, scale_each=True)
            
            heatmap = grid[0].cpu().numpy()
            heatmap_colored = cm.viridis(heatmap)[:, :, :3]
            wandb.log({f"Output/{layer_name}": wandb.Image(heatmap_colored, caption=f"Mean Output of {layer_name}"), "epoch": epoch})

    def log_eigen_featuremap(self, layer, layer_name, inputs, epoch):
        
        #Get Weights and Flatten: (Out, In, H, W) -> (K, C*H*W)
        w = layer.weight.data
        k, c, h, kw = w.shape
        w_flat = w.reshape(k, -1)

        if w_flat.shape[1] > 5000:
            return

        # Compute Gram Matrix and Eigen Decomposition
        gram = torch.matmul(w_flat.t(), w_flat)

        try:
            vals, vecs = torch.linalg.eigh(gram)
        except RuntimeError:
            return 

        num_vecs = min(10, vecs.shape[1])
        eigen_filters = vecs[:, -num_vecs:].t().reshape(num_vecs, c, h, kw)

        with torch.no_grad():
            out = F.conv2d(inputs, eigen_filters, stride=layer.stride, padding=layer.padding)
        '''
        if self.non_critical:
            grid = vutils.make_grid(out.reshape(out.shape[0] * num_vecs, 1, *out.shape[2:]), nrow=num_vecs, normalize=True, scale_each=True)
            wandb.log({f"Eigen_Filters/{layer_name}": wandb.Image(grid, caption=f"Top {num_vecs} Eigen Filter Responses"), "epoch": epoch})
        '''
        avg_out = out.mean(dim=1, keepdim=True)
        grid_avg = vutils.make_grid(avg_out, nrow=self.secondary_dim, normalize=True, scale_each=True)
        
        heatmap = grid_avg[0].cpu().numpy()
        heatmap_colored = cm.viridis(heatmap)[:, :, :3]
        wandb.log({f"Eigen_Featuremap_Mean/{layer_name}": wandb.Image(heatmap_colored, caption="Average Eigen Filter Response"), "epoch": epoch})

    def log_grad_cam(self, net, layer, layer_name, epoch, pred_targets=None):
        if self.viz_inputs is None or net is None or pred_targets is None:
            return

        inputs = self.viz_inputs
        targets = self.viz_targets
        
        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        try:
            layer_gc = LayerGradCam(net, layer)
            
            # 1. GradCAM wrt Predictions
            attr_pred = layer_gc.attribute(inputs, target=pred_targets, relu_attributions=True)
            attr_pred = LayerAttribution.interpolate(attr_pred, inputs.shape[2:], interpolate_mode='bilinear')
            
            # 2. GradCAM wrt Actual Targets
            attr_true = None
            if targets is not None:
                attr_true = layer_gc.attribute(inputs, target=targets, relu_attributions=True)
                attr_true = LayerAttribution.interpolate(attr_true, inputs.shape[2:], interpolate_mode='bilinear')

            def log_viz(attr, suffix, title):
                grid_img = vutils.make_grid(inputs, nrow=self.secondary_dim, normalize=True)
                grid_attr = vutils.make_grid(attr, nrow=self.secondary_dim, normalize=True, scale_each=True)
                
                img_np = grid_img.permute(1, 2, 0).cpu().numpy()
                attr_np = grid_attr.permute(1, 2, 0).cpu().numpy()
                
                heatmap_colored = cm.viridis(attr_np[:, :, 0])[:, :, :3]
                blended = 0.5 * img_np + 0.5 * heatmap_colored
                wandb.log({f"GradCAM_{suffix}/{layer_name}": wandb.Image(blended, caption=f"GradCAM {title} {layer_name}"), "epoch": epoch})

            log_viz(attr_pred, "Pred", "(Pred)")
            if attr_true is not None:
                log_viz(attr_true, "True", "(True)")
        finally:
            torch.set_grad_enabled(prev_grad_state)

    def call(self, layer, layer_name, activation, layer_input, epoch, net=None, pred_targets=None):

        viz_batch_size = min(len(layer_input), self.max_images)
        inputs_subset = layer_input[:viz_batch_size]

        self.log_featuremap(layer_name, activation, epoch)
        self.log_grad_cam(net, layer, layer_name, epoch, pred_targets=pred_targets)

        if self.non_critical:
            self.log_weights(layer, layer_name, epoch)
            self.log_eigen_featuremap(layer, layer_name, inputs_subset, epoch)
        
        