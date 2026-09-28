from torchmetrics import Metric
import torch 

class RGCH_Metric(Metric):
    """ 
    Unnormalised "raw" gradient correlation history store.
    """
    def __init__(self, header: list[str], **kwargs):
        super().__init__(**kwargs)
        _ = [
            self.add_state(f"{h}", 
                default=[], 
                dist_reduce_fx="cat" # if there was dist metrics used this is used
            ) 
            for h in header
        ]
        self._header = header

    def update(self, ordered_gc_vec: torch.Tensor) -> None:
        """ 
        Expects a gradient correlation vector for a timestep, ordered by the original header list passed in
        Tensor will have their state detached from autograd if not already. 
        Each element of vector has a name in header list, used for appended to corresponding state
        """
        for i, val in enumerate(ordered_gc_vec):
            name = self._header[i]
            self.state_dict()[name] = val
    
class GCH_Metric(RGCH_Metric):
    """
    Normalised gradient correlation history store.
    """
    def __init__(self, header: list[str], **kwargs):
        super().__init__(header, **kwargs)
        self.plot_lower_bound = -1 # Bounds are applicable here
        self.plot_upper_bound = 1

    def update(self, ordered_gc_vec: torch.Tensor) -> None:
        """ 
        Expects a gradient correlation vector for a timestep, ordered by the original header list passed in.
        Tensor will have their state detached from autograd if not already. 
        """