import torch.nn as nn
from typing import Any, Dict, Tuple

__all__ = ["get_dist_Muon_Adam_params", "get_dual_opt_params", "get_Adam_params", "get_nondist_Muon_Adam_params"]

def get_dist_Muon_Adam_params(model: Any, lr_muon: float, lr_adam: float, w_decay_muon: float = 0, w_decay_adam: float = 0, momentum_muon: float = 0.2, beta_1_adam: float = 0.95, beta_2_adam: float = 0.99, use_layer_recs: bool = True, eps: float = 1e-10) -> list[dict[str, Any]]:
    """
    For a list [m, a] m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    low_dim_params = [p for p in model.parameters() if len(p.shape) != 2 and p.requires_grad]
    high_dim_params = [p for n, p in model.named_parameters() if (len(p.shape) == 2) and (p.requires_grad)]
    
    # TODO: Differentiate Keller recs for 2nd block to test performance against it
    param_groups = [
        dict(params=high_dim_params, use_muon=True,
            lr=lr_muon, weight_decay=w_decay_muon, momentum=momentum_muon),
        dict(params=low_dim_params, use_muon=False,
            lr=lr_adam, betas=(beta_1_adam, beta_2_adam), weight_decay=w_decay_adam),
    ]

    return param_groups

def get_nondist_Muon_Adam_params(model: Any, lr_muon: float, lr_adam: float, use_muon:bool=True, w_decay_muon: float = 0, w_decay_adam: float = 0, momentum_muon: float = 0.2, beta_1_adam: float = 0.95, beta_2_adam: float = 0.999, use_layer_recs: bool = True, eps: float = 1e-8, nesterov:bool = False,) -> list[list[dict[str, Any]]]:
    """
    For a list [m, a] m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    low_dim_params = [p for p in model.parameters() if len(p.shape) == 1 and p.requires_grad]
    conv_params = [p for n, p in model.named_parameters() if (len(p.shape) == 4) and (p.requires_grad) and ("conv" in n)]
    high_dim_params = [p for n, p in model.named_parameters() if (p not in set(conv_params)) and (p not in set(low_dim_params))] # use set NOT list NOR unchanged tensor - shape error is thrown

    # TODO: Differentiate Keller recs for 2nd block to test performance against it
    if use_muon:
        param_groups = [
            dict(params=high_dim_params, use_muon=True,
                lr=lr_muon, weight_decay=w_decay_muon, momentum=momentum_muon),
            dict(params=conv_params, use_muon=True,
                lr=lr_muon, weight_decay=w_decay_muon, momentum=momentum_muon),
            dict(params=low_dim_params, use_muon=False,
                lr=lr_adam, betas=(beta_1_adam, beta_2_adam), weight_decay=w_decay_adam),
        ]
    else: 
        param_groups = [
            dict(params=high_dim_params, use_muon=False,
                lr=lr_adam, betas=(beta_1_adam, beta_2_adam), weight_decay=w_decay_adam, eps=eps),
            dict(params=conv_params, use_muon=False,
                lr=lr_adam, betas=(beta_1_adam, beta_2_adam), weight_decay=w_decay_adam, eps=eps),
            dict(params=low_dim_params, use_muon=False,
                lr=lr_adam, betas=(beta_1_adam, beta_2_adam), weight_decay=w_decay_adam, eps=eps),
        ]   
        
    return [param_groups]

def get_dual_opt_params(model: Any, lr_muon: float, lr_adam: float, w_decay_muon: float = 0, w_decay_adam: float = 0, momentum_muon: float = 0.2, beta_1_adam: float = 0.95, beta_2_adam: float = 0.99, nesterov:bool = False, use_layer_recs: bool = True, eps: float = 1e-10, use_muon: bool = True) -> list[list[dict[str, Any]]]:
    """
    For a list [m, a] m sets Muon's and a Adam's parameters, respectively.  
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors
    low_dim_params = [p for p in model.parameters() if len(p.shape) != 2 and p.requires_grad]
    high_dim_params = [p for p in model.parameters() if (len(p.shape) == 2) and (p.requires_grad)]

    if use_muon == True:
        muon_groups = [dict(params=high_dim_params, lr=lr_muon, weight_decay=w_decay_muon, momentum=momentum_muon, nesterov=False)]
        adam_groups = [dict(params=low_dim_params)]
        adam_groups = [dict(**g, betas=(beta_1_adam, beta_2_adam), eps=eps, lr=lr_adam, weight_decay=w_decay_adam) for g in adam_groups]
    
        return [muon_groups, adam_groups]
    else: 
        # This existed only for debugging TODO: Remove and put more graceful error handling and assurance 
        adam_groups = [dict(params=low_dim_params), dict(params=high_dim_params)]
        adam_groups = [dict(**g, betas=(beta_1_adam, beta_2_adam), eps=eps, lr=lr_adam, weight_decay=w_decay_adam) for g in adam_groups]
    
        return [adam_groups, adam_groups]

# TODO: Name these to be Adam native parameters not accepting same args as Muon
def get_Adam_params(model: Any, lr: float = 0.001, weight_decay: float = 0, beta_1: float = 0.95, beta_2: float = 0.99, eps: float = 1e-8) -> list[list[dict[str, Any]]]:
    """
    Use to get Adam native parameters   
    """
    # Norm biases are not innately 2D, but a collation of 1D vectors

    all_params = [p for p in model.parameters() if p.requires_grad]
    
    adam_groups = [dict(params=all_params)]
    adam_groups = [dict(**g, betas=(beta_1, beta_2), eps=eps, lr=lr, weight_decay=weight_decay) for g in adam_groups]
    
    return [adam_groups]