from typing import Callable, Type, Tuple, AnyStr, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from hydra_zen.typing import Partial
from hydra_zen import instantiate
from torchmetrics import MaxMetric, MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy

__all__ = ["ClassImbLitModule"]


class ClassImbLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        # TODO: Allow params to be either this OR strict tensor format for other use cases, may not fit streamlined MuonAuxAdam format  
        params: list[dict[str, Any]],
        optim: Partial[Optimizer],
        # scheduler: LRScheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.optim = optim
        self.params = params
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.max_val_acc = MaxMetric()

        self.model = model

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.max_val_acc.reset()

    def forward(self, input):
        return self.model(input)

    # TODO: Alter ordered tuple output to be a dictionary or leave it if input to output pipeline can handle it safely
    def batch_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        input, targets = batch
        logits = self.forward(input)
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
        metrics = self.batch_step(batch)
        loss, _, _ = metrics
        self.update_log_metrics(self.train_loss, self.train_acc, "train", metrics)
        # return loss or backpropagation will fail
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        metrics = self.batch_step(batch)
        self.update_log_metrics(self.val_loss, self.val_acc, "val", metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        metrics = self.batch_step(batch)
        self.update_log_metrics(self.test_loss, self.test_acc, "test", metrics)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

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


    def configure_optimizers(self) -> Optimizer:
        optimizer = self.optim(self.params)
        # TODO: Can try add configured learning rate scheduler as in Lhydra template, or check how MuonWithAuxAdam package implements it
        return optimizer

