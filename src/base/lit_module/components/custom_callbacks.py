import time
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch 
import numpy as np

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric

__all__ = ["StoreAllGCH", "TimerCallback", "DebugCallback", "StoreClassRatiosCallback"]

class StoreAllGCH(Callback):
    def __init__(self, what="CH", verbose=True):
        self.what = what
        self.verbose = verbose
        self.steps = 0
        self.state = dict(prev_param_grads = [])

    @property
    def state_key(self) -> str:
        # @lightning.ai :: note: we do not include `verbose` here on purpose
        return f"Counter[what={self.what}]"

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not trainer.training:
            return 
        
        self.trainable_params = [p for p in pl_module.parameters() if p.requires_grad]
        
        curr_param_grads = []
        prev_grads_exist = len(self.state["prev_param_grads"]) != 0
        curr_gcv = torch.Tensor(size=(len(self.trainable_params),))
        curr_r_gcv = torch.zeros_like(curr_gcv)

        for i, p in enumerate(self.trainable_params):
            assert p.grad is not None, "None gradients detected, debug optimizer/s invoked on current step"

            curr_p_grad = p.grad.detach().clone()
            curr_param_grads.append(curr_p_grad) 

            # Do not compute dot product if record does not exist
            if prev_grads_exist:
                prev_p_grad = self.state["prev_param_grads"][i].detach().clone()
                dot_prod = torch.sum(torch.mul(prev_p_grad, curr_p_grad))
                norm = prev_p_grad.data.norm() * curr_p_grad.data.norm()
                curr_gcv[i] = dot_prod / norm
                curr_r_gcv[i] = dot_prod

        # Current grad becomes past grads
        self.state["prev_param_grads"] = curr_param_grads 

        # curr_gcs (and curr_r_gcs) is filled only after a past and prev pair exist
        # Given the above, the global stores can be filled
        if trainer.global_step != 0: 
            assert type(pl_module.metric_gch) is GCH_Metric
            pl_module.metric_gch.update(curr_gcv, trainer.global_step, trainer.current_epoch)
            
            assert type(pl_module.metric_rgch) is RGCH_Metric
            pl_module.metric_rgch.update(curr_r_gcv, trainer.global_step, trainer.current_epoch)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

class DebugCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("\n=== CALLBACK IS ACTIVE ===\n")

# Credit to @HaukurPall from pytorch-lightning GH discussions
class TimerCallback(Callback):
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx) -> None:
        self.train_start_time = time.perf_counter() 

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
        self.elapsed = time.perf_counter() - self.train_start_time
        pl_module.log(
            "train/forward_time_seconds", 
            self.elapsed, 
            rank_zero_only=True, 
            on_step=True
        )

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss):
        self.backward_start = time.perf_counter()

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        backward_time = time.perf_counter() - self.backward_start
        pl_module.log(
            "train/backward_time_seconds",
            backward_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx=0):
        self.val_batch_start = time.perf_counter()

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx=0
        ):
        batch_time = time.perf_counter() - self.val_batch_start
        pl_module.log(
            "validation/forward_time_seconds",
            batch_time,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        self.step_start = time.perf_counter()

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Any) -> None:
        # This will get called at the beginning of training to clear any gradients from tuning etc.
        # In those cases the step_start is not set so we do nothing.
        if not hasattr(self, "step_start"):
            return
        step_end = time.perf_counter() - self.step_start
        pl_module.log(
            "train/step_time_seconds",
            step_end,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True  # no need to log for all devices
        )

class StoreClassRatiosCallback(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        raw_class_ratios = getattr(pl_module.datamodule, "class_ratios") 
        class_labels = getattr(pl_module.datamodule, "class_labels") 
        assert type(raw_class_ratios) is dict[str, int], f"Raw class ratios {raw_class_ratios} is not of required type"
        assert type(class_labels) is list[str], f"Class labels {class_labels} is not of required type"

        labelled_ratios = {k: v for k, v in zip(class_labels, raw_class_ratios.values())}
        assert type(labelled_ratios) is dict[str, int], f"Labelled class ratios {labelled_ratios} is not of required type"
        
        pl_module.log_dict(labelled_ratios, rank_zero_only=True)