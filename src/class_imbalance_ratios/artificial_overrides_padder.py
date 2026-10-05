__all__ = ["ArtificialOverridesPadder"]

from src.class_imbalance_ratios.conf_map_store import ConfMapStore


class ArtificialOverridesPadder():
    """ 
    This class is entirely made up to reduce workload, hence the "Artifical" prefix.  
    Input is dict of keys AND their values as expected in the VALID strict OmegaConf overrides 
    syntax and types bubbled up to Hydra-zen. 
    It creates something like a map (in the "grid of values" sense) with key labels
    of experiment config values that match. 
    """
    def __init__(self, overrides: dict[str,  float | int | str], padding_exceptions: list[str], pad_num: int) -> None:
        self._autofill_exceptions = padding_exceptions
        self._pad_num = pad_num
        self._overrides = self._padded_form(overrides)  

    def _padded_form(self, overrides: dict[str,  float | int | str]) -> dict[str, list[str]]: 
        """ 
        Returns override map for internal use and typing. 
        It repeats values in a python list of the initial OmegaConf syntax.
        """
        padded_overrides = {
            k: self._repeat_join_str(v) 
            for k, v in overrides.items()
            if k not in self._autofill_exceptions 
        }
        exceptions = {
            k: v
            for k, v in overrides.items()
            if k in self._autofill_exceptions 
        }

        merged = {**padded_overrides, **exceptions} 

        return merged
    
    def _repeat_join_str(self, val: str | int | float) -> list[str]:
        val = str(val) if type(val) is int else val
        string = [f"{val}"]
        string *= self._pad_num
        string = string

        return string   

    @property 
    def overrides_map(self) -> dict[str, list[str]]:
        return self._overrides

    @property 
    def overrides_map_as_store(self) -> ConfMapStore:
        return ConfMapStore(self.overrides_map, self.pad_constant)

    @property
    def pad_constant(self) -> int:
        return self._pad_num
