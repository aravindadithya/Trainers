import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.amp import autocast
import io
import os
import matplotlib.cm as cm
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients
import numpy as np


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

            # Sort inputs and targets by class label
            indices = torch.argsort(targets)
            inputs = inputs[indices]
            targets = targets[indices]

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

    def log_predictions_table(self, net, log_key, outputs_precomputed=None, extra_visuals=None):
        net.eval()
        inputs, targets = self.inputs, self.targets
        if inputs is None:
            return {}

        num_classes = self.config.get('num_classes', 10)
        columns = ["Image", "Ground Truth", "Prediction", "Confidence"] + [f"Score_{i}" for i in range(num_classes)]
        
        extra_keys = []
        if extra_visuals:
            extra_keys = list(extra_visuals.keys())
            columns.extend(extra_keys)

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

            for j in range(len(inputs_cpu)):
                img = inputs_cpu[j]
                #TODO: Handle the case when image is given as 1D tensor properly.
                if img.dim() == 1:
                    img = img.view(1, 28, 28)
                
                img_viz = vutils.make_grid(img, normalize=True)
                row = [wandb.Image(img_viz), str(targets_cpu[j].item()), str(preds_cpu[j].item()), confidences_cpu[j].item()]
                row.extend(probs_cpu[j].tolist())
                
                if extra_visuals:
                    for key in extra_keys:
                        if j < len(extra_visuals[key]):
                            row.append(extra_visuals[key][j])
                        else:
                            row.append(None)

                table.add_data(*row)

        if table.data:
            return {log_key: table}
        
        return {}

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
            wandb.log({**self.log_predictions_table(net, "Validation Predictions"), "epoch": epoch})
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

        all_logs = {}
        
        print("Generating Layer Visuals (GradCAM, IG, FeatureMaps)...")
        pred_targets = torch.argmax(outputs, dim=1)
        per_sample_visuals = {}

        for handler in layer_handlers.values():
            global_logs, sample_logs = handler.get_visuals(net=net, pred_targets=pred_targets)
            all_logs.update(global_logs)
            per_sample_visuals.update(sample_logs)

        # Use the pre-computed outputs to log the prediction table with extra visuals
        print("Logging Prediction Table...")
        all_logs.update(self.log_predictions_table(net, "Validation Predictions", outputs_precomputed=outputs, extra_visuals=per_sample_visuals))

        if all_logs:
            all_logs["epoch"] = epoch
            wandb.log(all_logs)
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

    def _create_blended_images(self, inputs, attrs):
        imgs = []
        inputs = inputs.detach().cpu()
        attrs = attrs.detach().cpu()
        
        for i in range(len(inputs)):
            img = inputs[i]
            attr = attrs[i]
            
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-7)
            attr_np = attr.squeeze().numpy()
            
            heatmap = cm.viridis(attr_np)[:, :, :3]
            blended = img_np * heatmap
            imgs.append(wandb.Image(blended))
        return imgs

    def _create_heatmap_images(self, attrs):
        imgs = []
        attrs = attrs.detach().cpu()
        for i in range(len(attrs)):
            attr = attrs[i]
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-7)
            attr_np = attr.squeeze().numpy()
            heatmap = cm.viridis(attr_np)[:, :, :3]
            imgs.append(wandb.Image(heatmap))
        return imgs

    def log_weights(self, layer, layer_name):
        """Computes Conv2d weight visualization."""
        w = layer.weight.data
        op, ip, q, s = w.shape
        w = w.reshape(op * ip, 1, q, s)
        if w.shape[0] > self.max_weight_filters:
            w = w[:self.max_weight_filters]
        grid_w = vutils.make_grid(w, nrow=ip, normalize=True, scale_each=False)
        return {f"Weight_Filters/{layer_name}": wandb.Image(grid_w, caption=f"Weights_Filter of {layer_name}")}

    def compute_featuremap_visuals(self, layer_name, activation):
        """Computes Conv2d activation visualization."""
        if activation is not None:
            avg_act = activation.mean(dim=1, keepdim=True)
            imgs = self._create_heatmap_images(avg_act)
            return {f"Output/{layer_name}": imgs}
        return {}

    def compute_eigen_featuremap_visuals(self, layer, layer_name, inputs):
        
        #Get Weights and Flatten: (Out, In, H, W) -> (K, C*H*W)
        w = layer.weight.data
        k, c, h, kw = w.shape
        w_flat = w.reshape(k, -1)

        if w_flat.shape[1] > 5000:
            return {}

        # Compute Gram Matrix and Eigen Decomposition
        gram = torch.matmul(w_flat.t(), w_flat)

        try:
            vals, vecs = torch.linalg.eigh(gram)
        except RuntimeError:
            return {}

        num_vecs = min(10, vecs.shape[1])
        eigen_filters = vecs[:, -num_vecs:].t().reshape(num_vecs, c, h, kw)

        with torch.no_grad():
            out = F.conv2d(inputs, eigen_filters, stride=layer.stride, padding=layer.padding)
            
        avg_out = out.mean(dim=1, keepdim=True)
        imgs = self._create_heatmap_images(avg_out)
        return {f"Eigen_Featuremap_Mean/{layer_name}": imgs}

    def compute_grad_cam_visuals(self, net, layer, layer_name):

        if not self.config:
            return {}

        visuals = {}
        inputs = self.viz_inputs
        
        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Optimization: Freeze model parameters to avoid computing gradients for weights
        original_requires_grad = {}
        for name, param in net.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False
        
        try:
            layer_gc = LayerGradCam(net, layer)
            num_classes = self.config.get('num_classes', 10)
            print(f"Computing GradCAM for {layer_name}...")

            for class_idx in range(num_classes):
                # GradCAM for current class
                attr = layer_gc.attribute(inputs, target=class_idx, relu_attributions=True)
                attr = LayerAttribution.interpolate(attr, inputs.shape[2:], interpolate_mode='bilinear')

                imgs = self._create_blended_images(inputs, attr)
                visuals[f"GradCAM-{class_idx}"] = imgs
        finally:
            # Restore requires_grad
            for name, param in net.named_parameters():
                param.requires_grad = original_requires_grad.get(name, True)
            torch.set_grad_enabled(prev_grad_state)
        
        return visuals

    def compute_ig_visuals(self, net, pred_targets=None):

        if not self.config or pred_targets is None:
            return {}

        inputs = self.viz_inputs

        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Optimization: Freeze model parameters to avoid computing gradients for weights
        original_requires_grad = {}
        for name, param in net.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        try:
            print("Computing Integrated Gradients...")
            ig = IntegratedGradients(net)
            attr = ig.attribute(inputs, target=pred_targets, n_steps=25, internal_batch_size=inputs.shape[0])
            # Sum absolute values across channels to get a single heatmap per image
            attr = torch.sum(torch.abs(attr), dim=1, keepdim=True)

            imgs = self._create_blended_images(inputs, attr)
            return {"IG(Predicted)": imgs}
        finally:
            # Restore requires_grad
            for name, param in net.named_parameters():
                param.requires_grad = original_requires_grad.get(name, True)
            torch.set_grad_enabled(prev_grad_state)
        
        return {}

    def update_layer_info(self, layer_name, layer, input, output):

        self.layer_info[layer_name] = {
            "layer": layer,
            "input": input,
            "output": output
        }

    def get_visuals(self, net=None, pred_targets=None):
        global_logs = {}
        per_sample_logs = {}
        
        last_layer_name = list(self.layer_info.keys())[-1] if self.layer_info else None
        
        if not self.layer_info:
            return global_logs, per_sample_logs

        for layer_name, info in self.layer_info.items():
            layer = info['layer']
            activation = info['output']
            layer_input = info['input']

            per_sample_logs.update(self.compute_featuremap_visuals(layer_name, activation))

            if self.non_critical:
                global_logs.update(self.log_weights(layer, layer_name))
                per_sample_logs.update(self.compute_eigen_featuremap_visuals(layer, layer_name, layer_input))

        if net and last_layer_name:
            last_layer = self.layer_info[last_layer_name]['layer']
            per_sample_logs.update(self.compute_grad_cam_visuals(net, last_layer, last_layer_name))
        
        if net and pred_targets is not None:
            per_sample_logs.update(self.compute_ig_visuals(net, pred_targets=pred_targets))
        
        return global_logs, per_sample_logs
