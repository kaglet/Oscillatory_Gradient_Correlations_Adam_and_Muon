__all__ = ["ArtificialOverridesPadder"]

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

        return padded_overrides
    
    def _repeat_join_str(self, val: str | int | float) -> list[str]:
        val = str(val) if type(val) is int else val
        string = [f"{val}"]
        string *= self._pad_num
        string = string

        return string   

    @property 
    def overrides_map(self) -> dict[str, list[str]]:
        return self._overrides

    def pad_constant(self) -> int:
        return self._pad_num
    
class ExperimentMapStore():
    """ 
    This store allows obtaining experiment values matched 
    at a requested index as a valid overrides dictionary. 
    It additionally allows overrides of a single key value.
    Additional extension may be a dict of keys and vals to override. 
    """
    def __init__(self, overrides_map: dict[str, list[str]], consistent_len: int) -> None:
        for item in overrides_map.items():
            if len(item) != consistent_len:
                ValueError(f"{item} in the overrides map of {len(item)} does not match assured consistent length of {consistent_len}")

        self._overrides_map = overrides_map
        self._consistent_len = consistent_len

    def overwrite(self, key: str, remap_vals: list[str]):
        """
        Overwrite the map at key with list of values. 
        """
        assert len(remap_vals) == self._consistent_len, f"Length of remap values of {len(remap_vals)} must match the length assured in map of {self._consistent_len}"

        try:
            setattr(self._overrides_map, key, remap_vals) 
        except:
            ValueError(f"{key} not found in current overrides map")

    def overwrite_many(self, subset_map: dict[str, list[str]]):
        """
        Overwrite the map at requested keys with new values. 
        """
        for k, v in subset_map.items():
            self.overwrite(k, v)
    
    def get_map_slice(self, map_idx:int) -> dict[str, str]:
        """ 
        Obtains a vertical slice 
        """
        overrides = self._overrides_map
        requested_slice = {k:v[map_idx] for k,v in overrides.items()}
        
        return requested_slice