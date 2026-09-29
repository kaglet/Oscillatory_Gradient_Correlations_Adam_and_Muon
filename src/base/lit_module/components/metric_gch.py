from torchmetrics import Metric
import torch 

class RGCH_Metric(Metric):
    """ 
    Unnormalised "raw" gradient correlation history store.
    """
    def __init__(self, header: list[str], name: str = "RGCH", **kwargs):
        super().__init__(**kwargs)
        _ = [
            self.add_state(f"{h}", 
                default=[], 
                persistent=True,
                dist_reduce_fx="cat" # if there was dist metrics used this is used
            ) 
            for h in header
        ]
        self._header = header
        self._name = name
        self.add_state("steps", [], persistent=True) # metadata state to track steps to epochs, not used in compute
        self.add_state("epochs", [], persistent=True)

    @property
    def name(self):
        return self._name
    
    def update(self, ordered_gc_vec: torch.Tensor, step: int, epoch: int) -> None:
        """ 
        Expects a gradient correlation vector for a timestep, ordered by the original header list passed in
        Tensor will have their state detached from autograd if not already. 
        Each element of vector has a name in header list, used for appended to corresponding state
        """
        for i, val in enumerate(ordered_gc_vec):
            name = self._header[i]
            layer_values = self.state_dict()[name]
            assert type(layer_values) is list, "Layer values must be a list for CH store"
            layer_values.append(val)

        steps = self.state_dict()["steps"]
        assert type(steps) is list, "Steps store must be a list for CH store"
        steps.append(torch.tensor(step, dtype=torch.int8))
        
        epochs = self.state_dict()["epochs"]
        assert type(epochs) is list, "Epochs store must be a list for CH store"
        epochs.append(torch.tensor(epoch, dtype=torch.int8))

    def compute(self) -> dict[str, list[torch.Tensor]]:
        """ 
        Returns a dict of each header name and its paired list of tensor scalar values
        """
        layer_dict = {h: self.state_dict()[h] for h in self._header}

        return layer_dict
    
class GCH_Metric(RGCH_Metric):
    """
    Normalised gradient correlation history store.
    """
    def __init__(self, header: list[str], **kwargs):
        super().__init__(header, **kwargs)
        self.plot_lower_bound = -1 # Bounds are applicable here
        self.plot_upper_bound = 1