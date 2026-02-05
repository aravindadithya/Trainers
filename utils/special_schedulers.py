import torch
import math

class CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=0.8):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay = decay

    def step(self, epoch=None):
        if self.T_cur + 1 == self.T_i:
             self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        super().step(epoch)

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]