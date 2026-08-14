import torch.nn as nn
from typing import Any, Dict, Tuple

__all__ = ["get_Muon_Aux_Adam_params", "get_Full_Adam_params"]

def get_Muon_Aux_Adam_params(model: Any, lr_each: Tuple[float, float], weight_decay_each: Tuple[float, float], betas: Tuple[float, float], use_layer_recs: bool = True ) -> list[dict[str, Any]]:
    """
    For a tuple (m, a) m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    # norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]        
    low_dim_params = [p for p in model.parameters() if len(p.shape) <= 1 and p.requires_grad]
    high_dim_params = [p for p in model.parameters() if len(p.shape) > 1 and p.requires_grad]
    
    # These are to optionally NOT use Muon in input and output layer for CNN
    # Due to nature of informal recommendation per Muon author Keller these are by default ignored for now 
    # TODO: Validate params and understand what blocks comprise and why input, output layers are not as Sequential > BasicBlock in definition of ResNet
    all_params = list(model.parameters())
    input_params = [all_params[0]]
    output_params = [all_params[-1]]

    if use_layer_recs:
        param_groups = [
            dict(params=high_dim_params, use_muon=True,
                lr=lr_each[0], weight_decay=weight_decay_each[0]),
            dict(params=low_dim_params, use_muon=False,
                lr=lr_each[1], betas=betas, weight_decay=weight_decay_each[1]),
        ]
    else: 
        param_groups = [
            dict(params=high_dim_params, use_muon=True,
                lr=lr_each[0], weight_decay=weight_decay_each[0]),
            dict(params=low_dim_params+norm_biases+input_params+output_params, use_muon=False,
                lr=lr_each[1], betas=betas, weight_decay=weight_decay_each[1]),
        ]

    return param_groups

def get_Full_Adam_params(model: nn.Module, lr: float, weight_decay: float, betas=Tuple[float, float]) -> list[dict[str, Any]]:
    all_params = list(model.parameters())

    # Return as param_groups instead of all parameters in the case own experiments over different Adam config is desired
    param_groups = [
        dict(params=all_params, use_muon=False,
            lr=lr, weight_decay=weight_decay)
    ]

    return param_groups