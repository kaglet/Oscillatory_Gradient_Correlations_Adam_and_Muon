from torchmetrics import Metric
import torch 
__all__ = ["RGCH_Metric", "GCH_Metric"]

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
            curr_list = getattr(self, name)
            assert type(curr_list) is list, "Layer values must be a list for CH store"

            curr_list.append(val)
            setattr(self, name, curr_list)

        assert type(self.steps) is list, "Steps must be list of GC metric"
        self.steps.append(torch.tensor(step, dtype=torch.int32))
        
        assert type(self.epochs) is list, "Epochs must be list of GC metric"
        self.epochs.append(torch.tensor(epoch, dtype=torch.int32))

    def compute(self) -> dict[str, list[torch.Tensor]]:
        """ 
        Returns a dict of each header name and its paired list of tensor scalar values
        """
        layer_dict = {h: getattr(self, h) for h in self._header}

        return layer_dict
    
class GCH_Metric(RGCH_Metric):
    """
    Normalised gradient correlation history store.
    """
    plot_lower_bound = -1
    plot_upper_bound = 1
    def __init__(self, header: list[str], **kwargs):
        super().__init__(header, **kwargs)

if __name__ == "__main__":
    pass