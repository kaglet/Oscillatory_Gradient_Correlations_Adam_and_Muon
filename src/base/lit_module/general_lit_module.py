from logging import log
import os
from typing import Callable, Dict, Iterator, Optional, Type, Tuple, AnyStr, Any
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

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
import torchvision
from src.base.lit_module.components.custom_grad_accum import GradAccumSchedule
import matplotlib.pyplot as plt

__all__ = ["DualOptLitModule", "DistOptLitModule", "SingleOptLitModule"]

class DualOptLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[list[dict[str, Any]]],
        optims: list[Optimizer],
        schdlrs: Optional[list[LRScheduler]] = None,
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
        # print("Model id/reference in memory is:", id(model))
        self._grad_accum_schedule = grad_accum_schedule

        self.optims = optims
        self.params = params
        # Crucial dual opt fixes/implementation from @shivvor2
        self.automatic_optimization = False # optim.step() must be called manually seemingly
        self._optimizer_step_count = 0

        # TODO: DO NOT leave this as default without it being set for specific stages
        self._init_grad_corr_stores()
    #     self._loggers = [
    #         CSVLogger(name="train_acc_steps"),
    #         CSVLogger(name="train_loss_steps"),
    #         CSVLogger(name="val_acc_steps"), 
    #         CSVLogger(name="val_loss_steps"),

    #         CSVLogger(name="train_acc_epoch"),
    # CSVLogger(name="train_loss_epoch"),
    # CSVLogger(name="val_acc_epoch"), 
    # CSVLogger(name="val_loss_epoch")]

        self.save_hyperparameters(logger=False, ignore=["model"])

    def _init_grad_corr_stores(self) -> bool:
        self.trainable_params = []
        self.global_grad_corr_store = []
        self.prev_grads = []
        self.per_param_names = []
        return True

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # By default lightning executes validation step sanity checks before training starts,
        # So it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.max_val_acc.reset()
        self._init_grad_corr_stores()

    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self) -> list[Optimizer]:
        # TODO: Add configured learning rate scheduler as in Lhydra template, or check how MuonWithAuxAdam package implements it, or from @shivvor2

        return self.optims

    def _store_grad_corr(self):
        if self.global_step == 0:
            # names_store = [n for (n,p) in self.trainable_params]
            # print("Not None grad store", names_store)
            return
        
        self.trainable_params = [(n, p) for n,p in self.model.named_parameters() if p.grad is not None]
        
        curr_param_grads = []
        prev_grads_exist = len(self.prev_grads) != 0
        curr_grad_corr_store = []
         
        for i, (n, p) in enumerate(self.trainable_params):
            assert p.requires_grad == True, "None gradients detected when trainable params expected"
            if p.grad is not None:
                curr_p_grad = p.grad.detach().clone()
                curr_param_grads.append(curr_p_grad) 
                
                if prev_grads_exist:
                    # print("Prev grad i:", i, " With length: ", len(self.prev_grads), " Training params length: ", len(self.trainable_params))
                    prev_p_grad = self.prev_grads[i].detach().clone()
                    dot_prod = float(torch.sum(torch.mul(prev_p_grad, curr_p_grad)))
                    norm = float(prev_p_grad.data.norm() * curr_p_grad.data.norm())
                    curr_grad_corr_store.append(dot_prod / norm)
            else: 
                # print("p.grad is None detected on i: ", i)
                pass
        self.prev_grads = curr_param_grads 

        if len(curr_grad_corr_store) != 0: # Prev grads may exist here but grad corr would not have been created yet, can only started from step for accum
            self.global_grad_corr_store.append(curr_grad_corr_store)

        if len(self.global_grad_corr_store) >= 2:
            prev_store_len = len(self.global_grad_corr_store[-1])
            curr_store_len = len(self.global_grad_corr_store[-1])

            assert prev_store_len == curr_store_len, f"Mismatch between sizes {prev_store_len} and {curr_store_len} for batch {self.global_step}"
        
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
    def _batch_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, is_Train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        input, targets = batch
        if is_Train:
            logits = self.forward(input)
            loss = self.criterion(logits, targets.to(dtype=torch.long))

            grad_accum = self._get_current_grad_accum()
            scaled_loss = loss / grad_accum
            self.manual_backward(loss=scaled_loss)

            should_step = (batch_idx + 1) % grad_accum == 0

            if should_step:
                # print("Storing gradd corr from batch step")
                self._store_grad_corr()
                optims = self.optimizers()
                optims = optims if isinstance(optims, list) else [optims]

                for i, opt in enumerate(optims):
                    # Call global opt step only at the end to help hooks
                    if i == len(optims) - 1:
                        opt.step()  # This increments global_step # TODO: Check if this has been fixed in update of PL
                    else:
                        # Access underlying optimizer directly to avoid double-counting
                        opt.optimizer.step()
                        
                    opt.zero_grad()

                self._optimizer_step_count += 1
        else: 
            with torch.no_grad():
                logits = self.forward(input)
                loss = self.criterion(logits, targets.to(dtype=torch.long))

        preds = torch.argmax(logits, dim=1)

        return loss, preds, targets

    def _print_params_per_opt(self, opt: Any) -> None:
        print(opt.__class__.__name__)
        len_mu = 0
        len_ad = 0
        for g in opt.param_groups:
            if g["use_muon"] == True:
                len_mu += len(g["params"])
            elif g["use_muon"] == False:
                len_ad += len(g["params"])
        print("Lengths of groups for Adam, Muon, and total", len_ad, len_mu, len_ad+len_mu)

    def _log_standard_metrics(self, loss_store: MeanMetric, acc_store: Metric, metric_name: AnyStr, metrics: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        loss, preds, targets = metrics
        loss_store(loss)
        acc_store(preds, targets)
        
        metrics_dict = {
            f"{metric_name}/loss": loss,
            f"{metric_name}/acc": acc_store.compute()
        }
        if metric_name == "train":
            self.loggers[0].log_metrics(metrics_dict)
        elif metric_name == "val":
            self.loggers[1].log_metrics(metrics_dict)
        # self.log_dict(dictionary=metrics_dict, on_step=True, on_epoch=False, prog_bar=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        metrics = self._batch_step(batch, batch_idx, is_Train=True)
        loss, _, _ = metrics
        self._log_standard_metrics(self.train_loss, self.train_acc, "train", metrics)
        # return loss or backpropagation will fail
        x, y = batch

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        metrics = self._batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        self._log_standard_metrics(self.val_loss, self.val_acc, "val", metrics)
        return loss 
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        metrics = self._batch_step(batch, batch_idx, is_Train=False)
        loss, _, _ = metrics
        self._log_standard_metrics(self.test_loss, self.test_acc, "test", metrics)
        return loss

    def _log_GCH_store(self) -> None:
        print(f"Saving GCH of train at root")
        self.per_param_names = [n for (n,p) in self.model.named_parameters() if p.requires_grad]
        fields = self.per_param_names
        rows = np.array(self.global_grad_corr_store, np.float32)
        assert self.trainer.log_dir is not None, "Trainer log directory not set, cannot log GCH"
        log_dir = os.path.join(self.trainer.log_dir, 'GCH_Epoch')
        os.makedirs(log_dir, exist_ok=True)

        with open(log_dir + f'/{self.current_epoch}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)     # Write header
            writer.writerows(rows)      # Write data rows
        self._init_grad_corr_stores()
        

    def on_train_epoch_end(self) -> None:
        self._log_GCH_store()

    # TODO: Get best value after a batch is run for validation step NOT for epoch
    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute() # get current val acc
        self.max_val_acc(acc)  # auto update store with best so far val acc
        
        # Recall metric is reset by us with lightning after each epoch.
        # So log value computed at this moment, not a reference to Metric object.
        self.loggers[2].log_metrics({"val": float(self.max_val_acc.compute())})

        # self.log("val/acc_best", self.max_val_acc.compute(), sync_dist=True, prog_bar=True)

class SingleOptLitModule(DualOptLitModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[list[dict[str, Any]]],
        optims: list[Optimizer],
        schdlrs: Optional[list[LRScheduler]] = None,
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),  
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        num_classes: int = 10,
    ):
        super().__init__(model, params, optims, schdlrs, grad_accum_schedule, num_classes)

class DistOptLitModule(DualOptLitModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[list[dict[str, Any]]],
        optims: list[Optimizer],
        schdlrs: Optional[list[LRScheduler]] = None,
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),  
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        num_classes: int = 10,
    ):
        super().__init__(model, params, optims, schdlrs, grad_accum_schedule, num_classes)