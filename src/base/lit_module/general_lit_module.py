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
        self.global_gcs = []
        self.global_r_gcs = []
        self.prev_param_grads = []
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
        self.trainable_params = [p for p in self.model.parameters() if p.grad is not None]
        
        curr_param_grads = []
        prev_grads_exist = len(self.prev_param_grads) != 0
        curr_gcs = torch.Tensor(size=(len(self.trainable_params),))
        curr_r_gcs = torch.zeros_like(curr_gcs)

        for i, p in enumerate(self.trainable_params):
            assert p.grad is not None, "None gradients detected, debug optimizer/s invoked on current step"

            curr_p_grad = p.grad.detach().clone()
            curr_param_grads.append(curr_p_grad) 

            # Do not compute dot product if record does not exist
            if prev_grads_exist:
                prev_p_grad = self.prev_param_grads[i].detach().clone()
                dot_prod = torch.sum(torch.mul(prev_p_grad, curr_p_grad))
                norm = prev_p_grad.data.norm() * curr_p_grad.data.norm()
                curr_gcs[i] = dot_prod / norm
                curr_r_gcs[i] = dot_prod

        # Immediate past becomes current grads collected
        self.prev_param_grads = curr_param_grads 

        # curr_gcs (and curr_r_gcs) is filled only after a past and prev pair exist
        # Given the above, the global stores can be filled
        if len(curr_gcs) != 0: 
            self.global_gcs.append(np.asarray(curr_gcs, dtype=np.float32))
            self.global_r_gcs.append(np.asarray(curr_r_gcs, dtype=np.float32))

        # Sanity check
        if len(self.global_gcs) >= 2:
            prev_store_len = len(self.global_gcs[-1])
            curr_store_len = len(self.global_gcs[-1])

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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        input, targets = batch

        logits = self.forward(input)
        loss = self.criterion(logits, targets.to(dtype=torch.long))

        grad_accum = self._get_current_grad_accum()
        scaled_loss = loss / grad_accum
        self.manual_backward(loss=scaled_loss)

        should_step = (batch_idx + 1) % grad_accum == 0

        if should_step:
            preds = torch.argmax(logits, dim=1)
            metrics = dict(loss=loss, preds=preds, targets=targets)
            self._log_standard_metrics(self.train_loss, self.train_acc, stage, metrics)

            optims = self.optimizers()
            optims = optims if isinstance(optims, list) else [optims]

            for i, opt in enumerate(optims):
                # Call global opt step only at the end to help hooks
                if i == len(optims) - 1:
                    opt.step()  # This increments global_step # TODO: Check if this has been fixed in update of PL
                else:
                    # Access underlying optimizer directly to avoid double-counting
                    opt.optimizer.step()

                self._store_grad_corr()
                opt.zero_grad()

            self._optimizer_step_count += 1

        return loss

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

    def _log_standard_metrics(self, loss_store: MeanMetric, acc_store: Metric, stage: str, metrics: Dict[str,torch.Tensor]):
        loss_store(metrics["loss"])
        acc_store(metrics["preds"], metrics["targets"])
        
        metrics_dict = {
            f"{stage}/loss": loss_store.compute(),
            f"{stage}/acc": acc_store.compute()
        }

        # Choose different loggers, configured with different paths 
        if stage == "train":
            self.loggers[0].log_metrics(metrics_dict)
        elif stage == "val":
            self.loggers[1].log_metrics(metrics_dict)
        elif stage == "test":
            self.loggers[2].log_metrics(metrics_dict)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        metrics = self._batch_step(batch, batch_idx, stage="train")

        return metrics 
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        with torch.no_grad():
            metrics = self._batch_step(batch, batch_idx, stage="val")

        return metrics 
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        with torch.no_grad():
            metrics = self._batch_step(batch, batch_idx, stage="test")

        return metrics 

    def _log_GCH_store(self) -> None:
        self.per_param_names = [n for (n,p) in self.model.named_parameters() if p.requires_grad]
        fields = self.per_param_names
        rows = np.array(self.global_gcs, np.float32)

        # Required check to infer for join operation after
        assert self.trainer.log_dir is not None, "Trainer log directory not set, cannot log GCH"
        log_dir = os.path.join(self.trainer.log_dir, 'GCH_Epoch')

        os.makedirs(log_dir, exist_ok=True)

        with open(log_dir + f'/{self.current_epoch}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)     # Write header
            writer.writerows(rows)      # Write many data rows
        self._init_grad_corr_stores()
        
    def on_train_epoch_end(self) -> None:
        self._log_GCH_store()

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