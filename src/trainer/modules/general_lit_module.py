from typing import Callable, Dict, Iterator, Optional, Type, Tuple, AnyStr, Any

import csv
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer  
from torch.optim.lr_scheduler import LRScheduler
from hydra_zen.typing import Partial
from hydra_zen import instantiate
from torchmetrics import MaxMetric, MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy
from src.trainer.modules.components.custom_grad_accum import GradAccumSchedule

__all__ = ["NonDistLitModule", "DistLitModule", "SingleOptLitModule"]

class SingleOptLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[dict[str, Any]],
        optim: Partial[Optimizer],
        schdlrs: list[Partial[LRScheduler]],
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),  
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        num_classes: int = 10,
    ):
        super().__init__()
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        print("Model id/reference in memory is:", id(model))
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.max_val_acc = MaxMetric()

        self.model = model

        self.optim = optim
        self.params = params

        self.init_grad_corr_stores()
        self.save_hyperparameters(logger=False, ignore=["model"])

    def init_grad_corr_stores(self) -> bool:
        self.trainable_params = []
        self.global_grad_corr_store = []
        self.prev_grads = []
        self.per_param_names = []
        return True

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.max_val_acc.reset()
        self.init_grad_corr_stores()

    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self) -> Optimizer:
        optim = self.optim(self.params)

        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def store_grad_corr(self, trainable_params: list[tuple[str, nn.Parameter]], batch_idx: int):
        curr_param_grads = []
        prev_grads_exist = len(self.prev_grads) != 0
        curr_grad_corr_store = []

        for i, (n, p) in enumerate(trainable_params):
            assert p.grad is not None, "None gradients detected when trainable params expected"
            if p.grad is not None:
                curr_p_grad = p.grad.detach().clone()
                curr_param_grads.append(curr_p_grad) 
                
                if prev_grads_exist:
                    prev_p_grad = self.prev_grads[i].detach().clone()
                    dot_prod = float(torch.sum(torch.mul(prev_p_grad, curr_p_grad)))
                    norm = float(prev_p_grad.data.norm() * curr_p_grad.data.norm())
                    curr_grad_corr_store.append(dot_prod / norm)
                else:
                    self.per_param_names.append(n)

        self.prev_grads = curr_param_grads 

        if len(curr_grad_corr_store) != 0: # Prev grads may exist here but grad corr would not have been created yet, can only started from step for accum
            self.global_grad_corr_store.append(curr_grad_corr_store)

        if len(self.global_grad_corr_store) >= 2:
            prev_store_len = len(self.global_grad_corr_store[-1])
            curr_store_len = len(self.global_grad_corr_store[-1])

            assert prev_store_len == curr_store_len, f"Mismatch between sizes {prev_store_len} and {curr_store_len} for batch {batch_idx}"
        
        return 1
    
    # TODO: Alter ordered tuple output to be a dictionary or leave it if input to output pipeline can handle it safely
    def batch_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, is_Train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        input, targets = batch
        if is_Train:
            logits = self.model(input)
            loss = self.criterion(logits, targets)
        else: 
            with torch.no_grad():
                logits = self.model(input)
                loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets

    def update_log_metrics(self, loss_store: MeanMetric, acc_store: Metric, metric_name: AnyStr, metrics: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        loss, preds, targets = metrics
        loss_store(loss)
        acc_store(preds, targets)
        self.log(f"{metric_name}/loss", loss_store, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{metric_name}/acc", acc_store, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=True)
        loss, _, _ = metrics
        self.update_log_metrics(self.train_loss, self.train_acc, "train", metrics)
        # return loss or backpropagation will fail
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        print(loss)
        self.update_log_metrics(self.val_loss, self.val_acc, "val", metrics)
        return loss 
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        self.update_log_metrics(self.test_loss, self.test_acc, "test", metrics)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_fit_end(self) -> None:
        print("Saving Gradient Correlation Store at root")
        fields = self.per_param_names
        # _ = [print(len(step_corr_store)) for step_corr_store in self.global_grad_corr_store]
        print(f"train/size_grad_corr_store", len(self.global_grad_corr_store))

        rows = np.array(self.global_grad_corr_store, np.float32)
        with open('GCS.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)     # Write header
            writer.writerows(rows)      # Write data rows

    # TODO: Get best value after a batch is run for validation step NOT for epoch
    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        """
        acc = self.val_acc.compute() # get current val acc
        self.max_val_acc(acc)  # auto update store with best so far val acc
        
        # Recall metric is reset by us with lightning after each epoch.
        # So log value computed at this moment, not a reference to Metric object.
        self.log("val/acc_best", self.max_val_acc.compute(), sync_dist=True, prog_bar=True)


class DualOptimLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[list[dict[str, Any]]],
        optims: list[Partial[Optimizer]],
        schdlrs: list[Partial[LRScheduler]],
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),  
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        num_classes: int = 10,
    ):
        super().__init__()
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.max_val_acc = MaxMetric()

        self.model = model
        self._grad_accum_schedule = grad_accum_schedule

        self.optims = optims
        self.params = params
        # Crucial dual opt fixes/implementation from @shivvor2
        self.automatic_optimization = False # optim.step() must be called manually seemingly
        self._optimizer_step_count = 0

        # TODO: DO NOT leave this as default without it being set for specific stages
        self.init_grad_corr_stores()
        self.save_hyperparameters(logger=False, ignore=["model"])

    def init_grad_corr_stores(self) -> bool:
        self.trainable_params = []
        self.global_grad_corr_store = []
        self.prev_grads = []
        self.per_param_names = []
        return True

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.max_val_acc.reset()
        self.init_grad_corr_stores()

    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self) -> list[Optimizer]:
        # optims = self.optims if isinstance(self.optims, list) else [self.optims]
        optims = [optim(params) for optim, params in zip(self.optims, self.params)]
        # TODO: Add configured learning rate scheduler as in Lhydra template, or check how MuonWithAuxAdam package implements it, or from @shivvor2
        # mu_lr = optims[0].param_groups[0]["lr"]
        # ad_lr = optims[1].param_groups[0]["lr"]
        # ad_wd = optims[1].param_groups[0]["weight_decay"]

        # assert mu_lr == 1e-1, f"Muon lr of {mu_lr} not set to configured value"
        # assert ad_lr == 1e-5, f"Adam lr of {ad_lr} not set to configured value"
        # assert ad_wd == 0, f"Adam weight decay of {ad_wd} not set to configured value"

        return optims

    def store_grad_corr(self, trainable_params: list[tuple[str, nn.Parameter]], batch_idx: int):
        """
        
        """
        curr_param_grads = []
        prev_grads_exist = len(self.prev_grads) != 0
        curr_grad_corr_store = []

        for i, (n, p) in enumerate(trainable_params):
            assert p.grad is not None, "None gradients detected when trainable params expected"
            if p.grad is not None:
                curr_p_grad = p.grad.detach().clone()
                curr_param_grads.append(curr_p_grad) 
                
                if prev_grads_exist:
                    prev_p_grad = self.prev_grads[i].detach().clone()
                    dot_prod = float(torch.sum(torch.mul(prev_p_grad, curr_p_grad)))
                    norm = float(prev_p_grad.data.norm() * curr_p_grad.data.norm())
                    curr_grad_corr_store.append(dot_prod / norm)
                else:
                    self.per_param_names.append(n)

        self.prev_grads = curr_param_grads 

        if len(curr_grad_corr_store) != 0: # Prev grads may exist here but grad corr would not have been created yet, can only started from step for accum
            self.global_grad_corr_store.append(curr_grad_corr_store)

        if len(self.global_grad_corr_store) >= 2:
            prev_store_len = len(self.global_grad_corr_store[-1])
            curr_store_len = len(self.global_grad_corr_store[-1])

            assert prev_store_len == curr_store_len, f"Mismatch between sizes {prev_store_len} and {curr_store_len} for batch {batch_idx}"
        
        return 1
    
    def _get_current_grad_accum(self) -> int:
        """Get current gradient accumulation factor.

        Returns:
            The gradient accumulation factor for the current step.
            Uses grad_accum_schedule, and overrides Trainer value.
        """
        accum = self._grad_accum_schedule.get_accum(self._optimizer_step_count)
        # assert accum == 32, f"Accum {accum} from epoch 0 is not default 4"
        return accum
    
    # TODO: Alter ordered tuple output to be a dictionary or leave it if input to output pipeline can handle it safely
    def batch_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, is_Train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        input, targets = batch
        if is_Train:
            logits = self.forward(input)
            loss = self.criterion(logits, targets)

            grad_accum = self._get_current_grad_accum()
            scaled_loss = loss / grad_accum
            self.manual_backward(loss=loss)

            should_step = (batch_idx + 1) % grad_accum == 0

            if should_step:
                self.trainable_params = [(n, p) for n,p in self.model.named_parameters() if p.requires_grad]
                self.store_grad_corr(trainable_params=self.trainable_params, batch_idx=batch_idx)

                optims = self.optimizers()
                optims = optims if isinstance(optims, list) else [optims]

                for i, opt in enumerate(optims):
                    if i == 0:
                        opt.step()  # This increments global_step
                    else:
                        # Access underlying optimizer directly to avoid double-counting
                        opt.optimizer.step()
                    opt.zero_grad()

                self._optimizer_step_count += 1
        else: 
            with torch.no_grad():
                logits = self.forward(input)
                loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        # print(logits[:3, :])
        # print(preds[:3])
        # print(targets[:3])

        return loss, preds, targets

    def update_log_metrics(self, loss_store: MeanMetric, acc_store: Metric, metric_name: AnyStr, metrics: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        loss, preds, targets = metrics
        loss_store(loss)
        acc_store(preds, targets)
        self.log(f"{metric_name}/loss", loss_store, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{metric_name}/acc", acc_store, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=True)
        loss, _, _ = metrics
        self.update_log_metrics(self.train_loss, self.train_acc, "train", metrics)
        # return loss or backpropagation will fail
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        # print(loss)
        self.update_log_metrics(self.val_loss, self.val_acc, "val", metrics)
        return loss 
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        metrics = self.batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        self.update_log_metrics(self.test_loss, self.test_acc, "test", metrics)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_fit_end(self) -> None:
        print("Saving Gradient Correlation Store at root")
        fields = self.per_param_names
        # _ = [print(len(step_corr_store)) for step_corr_store in self.global_grad_corr_store]
        self.log(f"train/size_grad_corr_store", len(self.global_grad_corr_store))

        rows = np.array(self.global_grad_corr_store, np.float32)
        with open('GCS.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)     # Write header
            writer.writerows(rows)      # Write data rows

    # TODO: Get best value after a batch is run for validation step NOT for epoch
    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        """
        acc = self.val_acc.compute() # get current val acc
        self.max_val_acc(acc)  # auto update store with best so far val acc
        
        # Recall metric is reset by us with lightning after each epoch.
        # So log value computed at this moment, not a reference to Metric object.
        self.log("val/acc_best", self.max_val_acc.compute(), sync_dist=True, prog_bar=True)

class NonDistLitModule(DualOptimLitModule):
    def __init__(
        self,
        model: nn.Module,
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        params: list[list[dict[str, Any]]],
        optims: list[Partial[Optimizer]],
        schdlrs: list[Partial[LRScheduler]],
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),
        num_classes: int=10,
    ):
        super().__init__(model=model, schdlrs=schdlrs, num_classes=num_classes, grad_accum_schedule=grad_accum_schedule, params=params, optims=optims)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save scheduler states to checkpoint.

        Args:
            checkpoint: The checkpoint dictionary to save state into.
        """
        checkpoint["optimizer_step_count"] = self._optimizer_step_count
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load scheduler states from checkpoint.

        Note:
            Actual scheduler state restoration happens in on_fit_start()
            because schedulers don't exist yet at this point.

        Args:
            checkpoint: The checkpoint dictionary to load state from.
        """
        self._optimizer_step_count = checkpoint.get("optimizer_step_count", 0)

class DistLitModule(DualOptimLitModule):
    def __init__(
        self,
        model: nn.Module,
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        params: list[list[dict[str, Any]]],
        optims: list[Partial[Optimizer]],
        schdlrs: list[Partial[LRScheduler]],
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),
        num_classes: int=10,
    ):
        super().__init__(model=model, schdlrs=schdlrs, num_classes=num_classes, grad_accum_schedule=grad_accum_schedule, params=params, optims=optims)