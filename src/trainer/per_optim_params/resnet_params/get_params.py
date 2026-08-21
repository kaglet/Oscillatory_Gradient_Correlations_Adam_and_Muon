import torch.nn as nn
from typing import Any, Dict, Tuple

__all__ = ["get_dist_Muon_Adam_params", "get_nondist_Muon_Adam_params", "get_Adam_params"]

def get_dist_Muon_Adam_params(model: Any, lr_each: Tuple[float, float], weight_decay_each: Tuple[float, float], betas: Tuple[float, float], use_layer_recs: bool = True, eps: float = 1e-10) -> list[dict[str, Any]]:
    """
    For a tuple (m, a) m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    low_dim_params = [p for p in model.parameters() if len(p.shape) != 2 and p.requires_grad]
    high_dim_params = [p for n, p in model.named_parameters() if (len(p.shape) == 2) and (p.requires_grad)]
    
    # TODO: Differentiate Keller recs for 2nd block to test performance against it
    param_groups = [
        dict(params=high_dim_params, use_muon=True,
            lr=lr_each[0], weight_decay=weight_decay_each[0]),
        dict(params=low_dim_params, use_muon=False,
            lr=lr_each[1], betas=betas, weight_decay=weight_decay_each[1]),
    ]

    return param_groups

def get_nondist_Muon_Adam_params(model: Any, lr_each: Tuple[float, float], weight_decay_each: Tuple[float, float], betas: Tuple[float, float], use_layer_recs: bool = True, eps: float = 1e-8) -> list[list[dict[str, Any]]]:
    """
    For a tuple (m, a) m sets Muon's and a Adam's parameters, respectively.  
    """
    print("Model id/reference in memory is:", id(model))
    # Norm biases are not innately 2D, but a collation of 1D vectors
    low_dim_params = [p for p in model.parameters() if len(p.shape) != 2 and p.requires_grad]
    high_dim_params = [p for p in model.parameters() if (len(p.shape) == 2) and (p.requires_grad)]
    
    muon_groups = [dict(params=high_dim_params, lr=lr_each[0], weight_decay=weight_decay_each[0], momentum=betas[0], nesterov=False)]
    adam_groups = [dict(params=low_dim_params)]
    adam_groups = [dict(**g, betas=betas, eps=eps, lr=lr_each[1], weight_decay=weight_decay_each[1]) for g in adam_groups]
    
    return [muon_groups, adam_groups]

def get_Adam_params(model: Any, lr_each: Tuple[float, float], weight_decay_each: Tuple[float, float], betas: Tuple[float, float], use_layer_recs: bool = True, eps: float = 1e-8) -> list[dict[str, Any]]:
    """
    For a tuple (m, a) m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    print("Model id/reference in memory is:", id(model))

    all_params = [p for p in model.parameters() if p.requires_grad]
    
    adam_groups = [dict(params=all_params)]
    adam_groups = [dict(**g, betas=betas, eps=eps, lr=lr_each[1], weight_decay=weight_decay_each[1]) for g in adam_groups]
    
    return adam_groups