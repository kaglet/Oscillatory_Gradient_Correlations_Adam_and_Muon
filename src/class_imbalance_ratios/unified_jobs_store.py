

# class UnifiedJobsStore():
#     """ 
#     Unified serial job run store for Muon and Adam. Can later be separated as potential TODO. 
#     """
#     def __init__(self, downsampler_ratios: list[int] = [20, 10, 5, 1]) -> None:
#         self._downsampler_ratios = downsampler_ratios
#         self._job_num_each = len(self._downsampler_ratios)
#         self._adam_overrides = self._set_adam_overrides()
#         self._muon_overrides = self._set_muon_overrides()

#     def _repeat_join_str(self, val: str | int | float) -> list[str]:
#         val = str(val) if type(val) is int else val
#         string = [f"{val}"]
#         string *= self._job_num_each
#         string = string
#         return string   

#     def _set_adam_overrides(self): 
#         adam_overrides = {
#             "model_cache": self._repeat_join_str(cif10_lbl), 
#             "datamodule": self._repeat_join_str(cif10_lbl), 
#             "params_cache": self._repeat_join_str(mu_aux_ad_lbl), 
#             "optims_cache": self._repeat_join_str(mu_aux_ad_lbl), 
#             "lit_module": self._repeat_join_str(cif10_lbl),
#             "++num_runs": self._repeat_join_str(2),
#             "++params_cache.lr_muon": self._repeat_join_str(1e-1),
#             "++params_cache.lr_adam": self._repeat_join_str(1e-1),
#             "++params_cache.momentum_muon": self._repeat_join_str(0.9),
#             "++params_cache.beta_1_adam": self._repeat_join_str(0.95),
#             "++params_cache.beta_2_adam": self._repeat_join_str(0.999),
#             "++params_cache.nesterov": self._repeat_join_str("False"),
#             "++params_cache.use_layer_recs": self._repeat_join_str("False"),
#             "++params_cache.eps": self._repeat_join_str(1e-8),
#             "++datamodule.downsampler_ratio":self.downsampler_ratios,
#             "++trainer.limit_train_batches": self._repeat_join_str(200),
#             "++trainer.limit_val_batches": self._repeat_join_str(100),
#             "++trainer.limit_test_batches": self._repeat_join_str(100)
#         }

#         return adam_overrides

#     def _set_muon_overrides(self): 
#         muon_overrides = {
#         "model_cache":self._repeat_join_str(cif10_lbl), 
#         "datamodule":self._repeat_join_str(cif10_lbl), 
#         "params_cache":self._repeat_join_str(mu_aux_ad_lbl), 
#         "optims_cache":self._repeat_join_str(mu_aux_ad_lbl), 
#         "lit_module":self._repeat_join_str(cif10_lbl),
#         "++num_runs": self._repeat_join_str(2),
#         "++params_cache.lr_muon":self._repeat_join_str(1e-1),
#         "++params_cache.lr_adam":self._repeat_join_str(1e-1),
#         "++params_cache.use_muon":self._repeat_join_str("False"),
#         "++params_cache.momentum_muon":self._repeat_join_str(0.9),
#         "++params_cache.beta_1_adam":self._repeat_join_str(0.95),
#         "++params_cache.beta_2_adam":self._repeat_join_str(0.999),
#         "++params_cache.nesterov":self._repeat_join_str("False"),
#         "++params_cache.use_layer_recs":self._repeat_join_str("False"),
#         "++params_cache.eps":self._repeat_join_str(1e-8),
#         "++datamodule.downsampler_ratio":self.downsampler_ratios,
#         "++trainer.limit_train_batches":self._repeat_join_str(200),
#         "++trainer.limit_val_batches":self._repeat_join_str(100),
#         "++trainer.limit_test_batches": self._repeat_join_str(100)
#         }

#         return muon_overrides
    
#     def get_run_vals(self, idx: int = 0, which: Literal['muon', "adam"] = 'muon') -> dict[str, str]:
#         overrides = self._muon_overrides if which == "muon" else self._adam_overrides
#         valid_slice = {k:v[idx] for k,v in overrides.items()}

#         return valid_slice

#     @property
#     def job_num_each(self):
#         return self._job_num_each

#     @property
#     def downsampler_ratios(self):
#         return self._downsampler_ratios