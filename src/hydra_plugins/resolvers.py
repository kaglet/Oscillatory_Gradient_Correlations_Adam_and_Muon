from omegaconf import OmegaConf
from src.trainer.modules.components.custom_resnet import create_resnet_model

__all__ = ["register_resolvers"]

def register_resolvers():
    built_cls_imb_s_model = create_resnet_model
    OmegaConf.register_new_resolver('dummy_model', built_cls_imb_s_model)