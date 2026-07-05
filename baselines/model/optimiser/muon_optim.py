""" 
TODO: 
- Fix requirement of model through different composition.
     - Perhaps can fetch model from config, and fetch info on it simply.
- With config dynamic setting of nonhidden params assuming static transformer model only. 
     - Can fetch params from config class maybe using model type, after passing in what Muon needs, possibly inheriting 
     - See Muon GitHub CIFAR implementation for pro code potentially of param setting to borrow
     - Set lr, decay with config. 
"""
from muon import MuonWithAuxAdam
hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = MuonWithAuxAdam(param_groups)