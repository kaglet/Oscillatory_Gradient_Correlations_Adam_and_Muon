from logging import log
import os
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Type, Tuple, AnyStr, Any
import lightning
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
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import MaxMetric, CatMetric, Metric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
import torchvision
from src.base.lit_module.components.custom_grad_accum import GradAccumSchedule
import matplotlib.pyplot as plt

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy

__all__ = ["DualOptLitModule", "DistOptLitModule", "SingleOptLitModule"]

class DualOptLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        params: list[list[dict[str, Any]]],
        optims: list[Optimizer],
        schedulers: list[LRScheduler],
        grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({0: 1}),  
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        num_classes: int = 10,
        labels: list[str] = 
        ["airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"] #cifar10 labels
    ):
        super().__init__()
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.metric_train_acc = MyMultiClassAccuracy(labels=labels, name="train/acc")
        self.metric_train_acc.persistent(mode=True)

        self.metric_val_acc = MyMultiClassAccuracy(labels=labels, name="val/acc")
        self.metric_val_acc.persistent(mode=True)

        self.metric_test_acc = MyMultiClassAccuracy(labels=labels, name="test/acc")
        self.metric_test_acc.persistent(mode=True)

        self.metric_train_loss = MyLoss("train/loss")
        self.metric_train_loss.persistent(mode=True)

        self.metric_val_loss = MyLoss("val/loss")
        self.metric_val_loss.persistent(mode=True)

        self.metric_test_loss = MyLoss("test/loss")
        self.metric_test_loss.persistent(mode=True)

        self.model = model
        self._grad_accum_schedule = grad_accum_schedule

        self.optims = optims
        self.params = params
        self.my_lr_schdlrs = schedulers
        self.automatic_optimization = False 
        # Begin dual opt fixes/implementation from @shivvor2
        self._optimizer_step_count = 0

        self._per_param_names = [n for (n,p) in self.model.named_parameters() if p.requires_grad]
        
        self.metric_rgch = RGCH_Metric(header=self._per_param_names, name="RGCH")
        self.metric_gch = GCH_Metric(header=self._per_param_names, name="GCH")

        self.save_hyperparameters(logger=False, ignore=["model"])

    def _reset_gch_stores(self) -> bool:
        self.metric_rgch.reset() 
        self.metric_gch.reset()
        return True

    def on_fit_start(self) -> None:
        self.metric_val_acc.reset()
        self.metric_val_loss.reset()

        self.metric_train_acc.reset()
        self.metric_train_loss.reset()

    def on_test_start(self) -> None:
        self.metric_test_acc.reset()
        self.metric_test_loss.reset()
    
    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self):
        # TODO: Add configured learning rate scheduler as in Lhydra template, or check how MuonWithAuxAdam package implements it, or from @shivvor2
        
        return self.optims, self.my_lr_schdlrs
    
    def _get_current_grad_accum(self) -> int:
        """Get current gradient accumulation factor.

        Returns:
            The gradient accumulation factor for the current step.
            Uses grad_accum_schedule, and overrides Trainer value.
        """
        accum = self._grad_accum_schedule.get_accum(self._optimizer_step_count)
        # assert accum == 32, f"Accum {accum} from epoch 0 is not default 4"
        return accum

    def _batch_step_eval(
        self, inputs: torch.Tensor, targets: torch.Tensor, batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Obtain loss and other metrics for a step in eval mode."""
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        metrics = dict(loss=loss, preds=preds, targets=targets)
        if stage == "val":
            self._update_standard_metrics(self.metric_val_loss, self.metric_val_acc, stage, metrics)
        elif stage == "test": 
            self._update_standard_metrics(self.metric_test_loss, self.metric_test_acc, stage, metrics)

        return loss

    # TODO: Alter ordered tuple output to be a dictionary or leave it if input to output pipeline can handle it safely
    def _batch_step_train(
        self, inputs: torch.Tensor, targets: torch.Tensor, batch_idx: int, stage: str
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)

        grad_accum = self._get_current_grad_accum()
        scaled_loss = loss / grad_accum
        self.manual_backward(loss=scaled_loss)

        preds = torch.argmax(logits, dim=1)
        metrics = dict(loss=loss, preds=preds, targets=targets)
        self._update_standard_metrics(self.metric_train_loss, self.metric_train_acc, stage, metrics)

        should_step = (batch_idx + 1) % grad_accum == 0

        if should_step:
            optims = self.optimizers()
            optims = optims if isinstance(optims, list) else [optims]
            schdlrs = self.lr_schedulers()
            schdlrs = schdlrs if isinstance(schdlrs, list) else [schdlrs]

            for i, (opt, lr_schdlr) in enumerate(zip(optims, schdlrs)):
                # Only this needs check where type changes to set type | None, unlike optims
                assert lr_schdlr is not None, "LR Scheduler, is None" 
                # Call global opt step only at the end to help hooks
                if i == len(optims) - 1:
                    opt.step()  # This increments global_step # TODO: Check if this has been fixed in update of PL
                    lr_schdlr.step(self.current_epoch)
                else:
                    # Access underlying optimizer directly to avoid double-counting
                    opt.optimizer.step()
                    lr_schdlr.step(self.current_epoch)

                # Before this, GCH callback will run
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

    def _update_standard_metrics(self, loss_store: MyLoss, acc_store: MyMultiClassAccuracy, stage: str, metrics: Dict[str,torch.Tensor]):
        loss_store.update(loss = metrics["loss"], step= self.trainer.global_step, epoch= self.trainer.current_epoch)
        acc_store.update(targets = metrics["targets"], preds=metrics["preds"], step= self.trainer.global_step, epoch=self.trainer.current_epoch)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        inputs, targets = batch
        metrics = self._batch_step_train(inputs=inputs, targets=targets.to(dtype=torch.long), batch_idx=batch_idx, stage="train")

        return metrics 
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set."""
        inputs, targets = batch

        with torch.no_grad():
            metrics = self._batch_step_eval(inputs=inputs, targets=targets.to(dtype=torch.long), batch_idx=batch_idx, stage="val")

        return metrics 
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set."""
        inputs, targets = batch

        with torch.no_grad():
            metrics = self._batch_step_eval(inputs=inputs, targets=targets.to(dtype=torch.long), batch_idx=batch_idx, stage="test")

        return metrics 

    def _save_metric(self, metric: Metric, root_dir: str, prefix: str, main_name: str) -> None: 
        log_path = Path(root_dir + f"/{prefix}_{main_name}.pt")
        torch.save(metric.state_dict(), log_path)

    def _save_train_metrics(self) -> None: 
        root_dir = self.trainer.default_root_dir 

        # Save train loss metric
        self._save_metric(
            metric=self.metric_train_loss,
            root_dir=root_dir, 
            main_name="loss", 
            prefix="train"
        )

        # Save train acc metric
        self._save_metric(
            metric=self.metric_train_acc,
            root_dir=root_dir, 
            main_name="acc", 
            prefix="train"
        )

        # Save rgch metric
        self._save_metric(
            metric=self.metric_rgch,
            root_dir=root_dir, 
            main_name=self.metric_rgch.name, 
            prefix="train"
        )

        # Save gch metric
        self._save_metric(
            metric=self.metric_gch,
            root_dir=root_dir, 
            main_name=self.metric_gch.name, 
            prefix="train"
        )

    def _save_val_metrics(self) -> None: 
        root_dir = self.trainer.default_root_dir 

        # Save val loss metric
        self._save_metric(
            metric=self.metric_val_loss,
            root_dir=root_dir, 
            main_name="loss", 
            prefix="val"
        )

        self._save_metric(
            metric=self.metric_val_acc,
            root_dir=root_dir, 
            main_name="acc", 
            prefix="val"
        )

    def _save_test_metrics(self) -> None: 
        root_dir = self.trainer.default_root_dir 
        self._save_metric(
            metric=self.metric_test_loss,
            root_dir=root_dir, 
            main_name="loss", 
            prefix="test"
        )

        self._save_metric(
            metric=self.metric_test_acc,
            root_dir=root_dir, 
            main_name="acc", 
            prefix="test"
        )

    def on_fit_end(self) -> None:   
        self._save_train_metrics()
        self._save_val_metrics()

    def on_test_end(self) -> None:
        self._save_test_metrics()

