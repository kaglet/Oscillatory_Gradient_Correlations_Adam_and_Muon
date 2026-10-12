class ConfMapStore():
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

        print("Key:\n", self._overrides_map, "\nvals:\t", remap_vals, "\nkey:\t", key)
        try:
            self._overrides_map[f"{key}"] = remap_vals
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