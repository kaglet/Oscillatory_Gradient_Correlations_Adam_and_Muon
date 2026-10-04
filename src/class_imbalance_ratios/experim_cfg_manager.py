__all__ = ["OverridesRunManager"]

class OverridesRunManager():
    """ 
    This manages only the independent experiments and value matching done to get appropriate combinations of values at the desired experiment index.
    Dict keys AND their values are expected in the VALID strict OmegaConf overrides syntax and types bubbled up to Hydra-zen. 
    Multi-run is NOT recommended to be set via this. Ensure any range of values is in pure list format `[]`.

    Exception to the typing of overrides is when passing a dict of python `list` of `str` values (not OmegaConf).
    Each `str` entry is therefore individually of the required OmegaCong type. 

    Unified serial job run store for Muon and Adam. Can later be separated as potential TODO. 
    This stores values for a series of experiments. It does not control running experiments multiple times at a meta level. 
    The `get_run_vals` function augments this functional need however. 
    """
    def __init__(self, overrides: dict[str,  float | int | str] | dict[str, list[str]], autofill_exceptions: list[str], num_experiments: int, run_idxs: list[int] = [0,1,2,3], autofill: bool = True) -> None:
        self._num_experiments = num_experiments
        self._experim_iter = iter(list(range(self._num_experiments)))
        self._run_iter = iter(run_idxs) # custom run index

        self._autofill_exceptions = autofill_exceptions
        if autofill_exceptions:
            self._overrides = self._set_overrides_for_experiments(overrides) 
        else: 
            self._overrides = self._set_overrides_for_experiments(overrides) 

    def _repeat_join_str(self, val: str | int | float) -> list[str]:
        val = str(val) if type(val) is int else val
        string = [f"{val}"]
        string *= self._num_experiments
        string = string

        return string   

    def _set_overrides_for_experiments(self, overrides: dict[str,  float | int | str] | dict[str, list[str]]) -> dict[str, list[str]]: 
        """ 
        Returns override map for internal use and typing. 
        It repeats values in a python list of the initial OmegaConf syntax.
        """
        appended_overrides = {
            k: self._repeat_join_str(v) 
            for k, v in overrides 
            if k not in self._autofill_exceptions # TODO: All will be set to exceptions if manually set in full i.e. autofill is off, then store must match in type, don't like dual dependencies but yeah 
        }

        return appended_overrides

    def advance_run(self) -> None:
        self._curr_run = next(self._run_iter)

    @property
    def get_curr_run(self) -> int:
        return self._curr_run

    def advance_experim(self):
        self._curr_experim = next(self._experim_iter)

    @property
    def get_curr_experim(self) -> int:
        return self._curr_experim
    
    def get_experim_vals(self) -> dict[str, str]:
        overrides = self._overrides
        valid_slice = {k:v[self._curr_experim] for k,v in overrides.items()}
        run_slice = {**valid_slice, "++run_idx":f"{str(self._curr_run)}"}
        
        return run_slice

    @property
    def job_num_each(self):
        return self._num_experiments