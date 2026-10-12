
""" 
Generic names may be used to label configs.
When registering configs under a group in the config store, 
in hydra or hydra-zen, they require a name unique only to 
the scope of the group. This acts as the identifier (ID). 

All config names exist under this registry. 
Sharing can occur across groups. Generic names 
below denote certain defaults that are more suitable to 
label a configs behaviour in the context of different possible labels. 

See the config here for more detail.
"""

from dataclasses import dataclass

__all__ = ["shared_config_names", ]

@dataclass
class SharedConfigIDs:
    """
    Configs may fit to having settings suited for a niche case or 
    any more generic case (can fit to anything).

    Source of configs is denoted by name. 
    `experiment` denotes that level of config composition comes from research paper experiment. 

    Standalone configs may get more unique names, if they are merely composed into some other body that gets another ID.   
    `adam`, `muon_auxilliary_adam` satisfy such. 
    

    There does exist a hierarchy of levels to better describe 
    which violations in hierarchy should not be mixed up. 
    
    `muon_auxilliary_adam` <-> `adam`. 
    TODO: A guide will be placed for when consistency is required. 

    More a specific example of implementation see `./run_with_cfg.py`
    """
    adam = "adam"
    experiment = "cls_imb_cif10" 
    muon = "muon"
    chpc = "chpc"
    local = "local"

shared_config_names = SharedConfigIDs()