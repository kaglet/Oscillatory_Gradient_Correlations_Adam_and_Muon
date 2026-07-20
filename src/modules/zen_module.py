from typing import Callable, Type, Tuple, AnyStr

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler
from hydra_zen.typing import Partial

from torchmetrics import MaxMetric, MeanMetric, Metric
from torchmetrics.classification.accuracy import Accuracy
from muon import MuonWithAuxAdam
from torch.optim import Adam, AdamW

__all__ = ["MNISTLitModule", "train_and_eval"]


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


    def configure_optimizers(self):
        # Implicitly equivalent to self.model.parameters()
        # TODO: Do if statement logic for now for Muon and Adam handling how to run optim
        # TODO: Parameters depend on architecture to formulate and select which may be messy, I want it model agnostic but it is not, e.g. select only layers for Muon trying to keep it neutral, and select everything else for Adam, or the other way around
        filter_params = [p for p in self.model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in self.model.named_parameters() if 'norm' in n and p.requires_grad]        
        input_weights = [*self.model.head.(), *self.model.embed.parameters()]
        adam_param_configs = [dict(params=[self.model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[self.model.head.weight], lr=head_lr, weight_decay=wd/head_lr),
                     dict(params=[*list(self.model.children())[:-1]], lr=head_lr, weight_decay=wd/head_lr)]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                lr=0.02, weight_decay=0.01),
            dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
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