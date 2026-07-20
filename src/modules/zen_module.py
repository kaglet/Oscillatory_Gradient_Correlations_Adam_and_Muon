from typing import Callable, Type, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler
from hydra_zen.typing import Partial

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

__all__ = ["MNISTLitModule", "single_layer_nn", "train_and_eval"]


class MNISTLitModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim: Partial[Optimizer],
        dataloader: Type[DataLoader],
        # scheduler: LRScheduler,
        target_fn: Callable[[torch.Tensor], torch.Tensor],
        training_domain: torch.Tensor,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.optim = optim
        self.dataloader = dataloader

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

        self.training_domain = training_domain
        self.target_fn = target_fn

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

    # TODO: Unify training, validation, test steps which repeat their processing
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, preds, targets = self.batch_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, preds, targets = self.batch_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.batch_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.max_val_acc(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.max_val_acc.compute(), sync_dist=True, prog_bar=True)


    def configure_optimizers(self):
        # Implicitly equivalent to self.model.parameters()
        # TODO: Do if statement logic for now for Muon and Adam handling how to run optim
        return self.optim(self.parameters())

    def train_dataloader(self):
        # generate dataset: input, cos(input)
        input = self.training_domain.reshape(-1, 1)
        targets = self.target_fn(input)
        return self.dataloader(TensorDataset(input, targets))

def train_and_eval(
    model: torch.nn.Module,
    optim: Partial[Optimizer],
    dataloader: Type[DataLoader],
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    training_domain: torch.Tensor,
    lit_module: Type[MNISTLitModule],
    trainer: pl.Trainer,
):


    lit = lit_module(
        model=model,
        optim=optim,
        dataloader=dataloader,
        target_fn=target_fn,
        training_domain=training_domain,
    )

    # train the model
    trainer.fit(lit)

    # evaluate the model over the domain to assess the fit
    final_eval = lit(training_domain.reshape(-1, 1))
    final_eval = final_eval.detach().cpu().numpy().ravel()

    # return the final evaluation of our model:
    # a shape-(N,) numpy-array
    return final_eval