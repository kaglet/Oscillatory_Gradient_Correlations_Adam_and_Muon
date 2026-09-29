from lightning.pytorch.callbacks import Callback
import torch 
import numpy as np

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

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if not trainer.training:
            return 
        
        self.trainable_params = [p for p in pl_module.parameters() if p.grad is not None]
        
        curr_param_grads = []
        prev_grads_exist = len(self.state["prev_param_grads"]) != 0
        curr_gcs = torch.Tensor(size=(len(self.trainable_params),))
        curr_r_gcs = torch.zeros_like(curr_gcs)

        for i, p in enumerate(self.trainable_params):
            assert p.grad is not None, "None gradients detected, debug optimizer/s invoked on current step"

            curr_p_grad = p.grad.detach().clone()
            curr_param_grads.append(curr_p_grad) 

            # Do not compute dot product if record does not exist
            if prev_grads_exist:
                prev_p_grad = self.state["prev_param_grads"][i].detach().clone()
                dot_prod = torch.sum(torch.mul(prev_p_grad, curr_p_grad))
                norm = prev_p_grad.data.norm() * curr_p_grad.data.norm()
                curr_gcs[i] = dot_prod / norm
                curr_r_gcs[i] = dot_prod

        # Current grad becomes past grads
        self.state["prev_param_grads"] = curr_param_grads 

        # curr_gcs (and curr_r_gcs) is filled only after a past and prev pair exist
        # Given the above, the global stores can be filled
        if len(curr_gcs) != 0: 
            pl_module.state_dict()["metric_gch"](curr_gcs, trainer.global_step, trainer.current_epoch)
            pl_module.state_dict()["metric_rgch"](curr_r_gcs, trainer.global_step, trainer.current_epoch)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
