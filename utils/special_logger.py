import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.amp import autocast
import io
import os
import matplotlib.cm as cm
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients


class BaseLogger:

    def __init__(self, config):
        
        self.config = config
        self.optimizer = config.get('optimizer')
        self.lfn = config.get('lfn')
        self.net = config.get('net')
        self.scheduler = config.get('scheduler')

        self._initialize_wandb(config)

        self.start_epoch = 1
        self.best_val_acc = 0
        self.best_val_loss = float("inf")
        self.best_state_dict = None
        self.rotate_inputs = config.get('rotate_inputs', True)
        self.max_images = config.get('max_images', 32)

        self.inputs, self.targets = self.get_viz_inputs(config.get('val_loader'))

        if wandb.run.resumed:
            self._resume_run()
        else:
            self._log_initial_artifacts(self.inputs)

    def _initialize_wandb(self, config):

        wandb.login(key=os.getenv('WANDB_API_KEY'))

        # Filter out non-serializable objects for WandB config
        exclude_keys = ['net', 'train_loader', 'val_loader', 'test_loader', 'optimizer', 'lfn', 'scheduler']
        wandb_config = {k: v for k, v in config.items() if k not in exclude_keys}

        wandb.init(project=config['project'], name=config['run_name'], resume="allow", id=config['run_id'], config=wandb_config, entity=config['entity'])

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

        

        '''
        for i in range(wandb_config.get('num_classes', 10)):
            wandb.define_metric(f"GradCAM/class_{i}", step_metric="epoch")
        
        for name, layer in net.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                wandb.define_metric(f"Weights_Images/{name}", step_metric="epoch")
                wandb.define_metric(f"Activation_Images/{name}", step_metric="epoch")
        '''
    def get_viz_inputs(self, val_loader):

        try:
            # Get num_classes from config, with a default.
            num_classes = self.config.get('num_classes')
            images_per_class = max(1, self.max_images // num_classes)
            target_num_images = images_per_class * num_classes

            class_counts = {i: 0 for i in range(num_classes)}
            collected_inputs, collected_targets = [], []

            # Iterate through the loader to find a balanced set of images
            for batch_inputs, batch_targets in val_loader:
                for i in range(len(batch_inputs)):
                    label = batch_targets[i].item()
                    if label in class_counts and class_counts[label] < images_per_class:
                        collected_inputs.append(batch_inputs[i])
                        collected_targets.append(batch_targets[i])
                        class_counts[label] += 1

                if sum(class_counts.values()) >= target_num_images:
                    break

            if not collected_inputs:
                print("Viz setup failed: Could not collect any images from val_loader.")
                return None, None

            print(f"Collected {len(collected_inputs)} images for visualization ({images_per_class} from each of {num_classes} classes).")
            inputs = torch.stack(collected_inputs)
            targets = torch.stack(collected_targets)

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
            return inputs, targets

        except Exception as e:
            print(f"Viz setup failed with an unexpected error: {e}")
            return None, None

    def _resume_run(self):
        print("Resuming from previous run...")
        self.best_val_acc = wandb.run.summary.get("best_val_accuracy", 0)
        self.best_val_loss = wandb.run.summary.get("best_val_loss", float("inf"))
        try:
            # Load checkpoint to resume training state
            artifact = wandb.use_artifact(f"checkpoint-{self.config['run_id']}:latest")
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
                best_model_artifact = wandb.use_artifact(f"model-{self.config['run_id']}:latest")
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
      
        # Log initial weights for new runs
        artifact = wandb.Artifact(f"init-weights-{self.config['run_id']}", type='model', metadata={"epoch": 0})
        with artifact.new_file('init_model.pth', mode='wb') as f:
            torch.save({'state_dict': self.net.state_dict()}, f)
        wandb.log_artifact(artifact)
        
        if inputs is not None:
            self.net.eval()
            dummy_input = inputs[0].unsqueeze(0).cuda()
            artifact = wandb.Artifact(f"onnx-{self.config['run_id']}", type='model')
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

    def log_predictions_table(self, net, epoch, log_key, outputs_precomputed=None):
        net.eval()
        inputs, targets = self.inputs, self.targets
        if inputs is None:
            return

        num_classes = self.config.get('num_classes', 10)
        columns = ["Image", "Ground Truth", "Prediction", "Confidence"] + [f"Score_{i}" for i in range(num_classes)]
        table = wandb.Table(columns=columns)
        
        with torch.no_grad():
            if outputs_precomputed is not None:
                outputs = outputs_precomputed
            else:
                with autocast(device_type='cuda'):
                    outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            
            confidences, preds = torch.max(probs, 1)
            
            if len(targets.size()) > 1:
                _, targets = torch.max(targets, -1)
            else:
                targets = targets

            # Move to CPU for faster iteration and logging
            inputs_cpu = inputs.cpu()
            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()
            confidences_cpu = confidences.cpu()
            probs_cpu = probs.cpu()

            # Limit to 64 images to prevent hanging on large visualization batches
            for j in range(min(len(inputs_cpu), 64)):
                img = inputs_cpu[j]
                #TODO: Handle the case when image is given as 1D tensor properly.
                if img.dim() == 1:
                    img = img.view(1, 28, 28)
                
                img_viz = vutils.make_grid(img, normalize=True)
                row = [wandb.Image(img_viz), str(targets_cpu[j].item()), str(preds_cpu[j].item()), confidences_cpu[j].item()]
                row.extend(probs_cpu[j].tolist())
                table.add_data(*row)

        if table.data:
            wandb.log({log_key: table, "epoch": epoch})

    def log_visuals(self, net, epoch):
        """
        Coordinates logging of all visuals for an epoch, including prediction tables
        and specialized visuals like Grad-CAM, feature maps, etc.
        This method performs a single forward pass and shares the results.
        """
        if self.inputs is None:
            return

        print("Generating Visuals...")
        net.eval()

        # This logic is moved from the global specialized_visuals_dispatcher
        layer_handlers = {}
        layer_handlers[torch.nn.Conv2d] = CNNLogger(self.inputs, self.targets, config=self.config, secondary_dim=4, max_weight_filters=20)

        # If no specialized handlers, just log the prediction table and return.
        if not layer_handlers:
            self.log_predictions_table(net, epoch, "Validation Predictions")
            return

        hooks = []
        def get_activation(name, layer, handler):
            def hook(model, input, output):
                inp = input[0] if isinstance(input, tuple) else input
                handler.update_layer_info(name, layer, inp.detach(), output.detach())
            return hook

        for name, layer in net.named_modules():
            if type(layer) in layer_handlers:
                handler = layer_handlers[type(layer)]
                hooks.append(layer.register_forward_hook(get_activation(name, layer, handler)))

        # Perform the single forward pass. Hooks will be executed here.
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = net(self.inputs)

        # Remove hooks immediately after use
        for h in hooks:
            h.remove()

        # Use the pre-computed outputs to log the prediction table
        print("Logging Prediction Table...")
        self.log_predictions_table(net, epoch, "Validation Predictions", outputs_precomputed=outputs)

        # Continue with specialized logging using the populated handlers
        print("Generating Layer Visuals (GradCAM, IG, FeatureMaps)...")
        pred_targets = torch.argmax(outputs, dim=1)
        for handler in layer_handlers.values():
            handler.log_all(epoch, net=net, pred_targets=pred_targets)
        print("Visuals Logging Completed.")

    def finish(self):
        wandb.finish()


class CNNLogger:

    def __init__(self, inputs, targets, secondary_dim=1 , max_weight_filters=100, config=None):
        self.max_weight_filters = max_weight_filters
        self.non_critical = os.getenv('NON_CRITICAL_LOGS', 'False').lower() in ('true', '1', 't')
        self.secondary_dim = secondary_dim
        self.viz_inputs = inputs
        self.viz_targets = targets
        self.layer_info = {}
        self.config = config

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
            
        avg_out = out.mean(dim=1, keepdim=True)
        grid_avg = vutils.make_grid(avg_out, nrow=self.secondary_dim, normalize=True, scale_each=True)
        
        heatmap = grid_avg[0].cpu().numpy()
        heatmap_colored = cm.viridis(heatmap)[:, :, :3]
        wandb.log({f"Eigen_Featuremap_Mean/{layer_name}": wandb.Image(heatmap_colored, caption="Average Eigen Filter Response"), "epoch": epoch})

    def log_grad_cam(self, net, layer, layer_name, epoch, pred_targets=None):
        if not self.config:
            return

        inputs = self.viz_inputs
        
        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        try:
            layer_gc = LayerGradCam(net, layer)
            num_classes = self.config.get('num_classes', 10)
            print(f"Computing GradCAM for {layer_name}...")

            for class_idx in range(num_classes):
                # GradCAM for current class
                attr = layer_gc.attribute(inputs, target=class_idx, relu_attributions=True)
                attr = LayerAttribution.interpolate(attr, inputs.shape[2:], interpolate_mode='bilinear')

                # Create visualization for this class
                grid_img = vutils.make_grid(inputs, nrow=self.secondary_dim, normalize=True)
                grid_attr = vutils.make_grid(attr, nrow=self.secondary_dim, normalize=True, scale_each=True)

                img_np = grid_img.permute(1, 2, 0).cpu().numpy()
                attr_np = grid_attr.permute(1, 2, 0).cpu().numpy()

                heatmap_colored = cm.viridis(attr_np[:, :, 0])[:, :, :3]
                blended = img_np * heatmap_colored

                wandb.log({f"GradCAM/class_{class_idx}": wandb.Image(blended, caption=f"GradCAM for Class {class_idx}"), "epoch": epoch})
        finally:
            torch.set_grad_enabled(prev_grad_state)

    def log_integrated_gradients(self, net, epoch, pred_targets=None):
        if not self.config or pred_targets is None:
            return

        inputs = self.viz_inputs

        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        try:
            print("Computing Integrated Gradients...")
            ig = IntegratedGradients(net)
            attr = ig.attribute(inputs, target=pred_targets, n_steps=25)
            # Sum absolute values across channels to get a single heatmap per image
            attr = torch.sum(torch.abs(attr), dim=1, keepdim=True)

            # Create visualization
            grid_img = vutils.make_grid(inputs, nrow=self.secondary_dim, normalize=True)
            grid_attr = vutils.make_grid(attr, nrow=self.secondary_dim, normalize=True, scale_each=True)

            img_np = grid_img.permute(1, 2, 0).cpu().numpy()
            attr_np = grid_attr.permute(1, 2, 0).cpu().numpy()

            heatmap_colored = cm.viridis(attr_np[:, :, 0])[:, :, :3]
            blended = img_np * heatmap_colored

            wandb.log({"IntegratedGradients/Predicted": wandb.Image(blended, caption="IG for Predicted Class"), "epoch": epoch})
        finally:
            torch.set_grad_enabled(prev_grad_state)

    def update_layer_info(self, layer_name, layer, input, output):
        self.layer_info[layer_name] = {
            "layer": layer,
            "input": input,
            "output": output
        }

    def log_all(self, epoch, net=None, pred_targets=None):
        last_layer_name = list(self.layer_info.keys())[-1] if self.layer_info else None
        for layer_name, info in self.layer_info.items():
            layer = info['layer']
            activation = info['output']
            layer_input = info['input']
            
            inputs_subset = layer_input

            self.log_featuremap(layer_name, activation, epoch)
            if layer_name == last_layer_name:
                self.log_grad_cam(net, layer, layer_name, epoch, pred_targets=pred_targets)
                self.log_integrated_gradients(net, epoch, pred_targets=pred_targets)

            if self.non_critical:
                self.log_weights(layer, layer_name, epoch)
                self.log_eigen_featuremap(layer, layer_name, inputs_subset, epoch)