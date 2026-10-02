from typing import Any

from torchmetrics import Metric
import torch 

"""
Still follow general convention of returning main desired result in compute etc., some reducible state beyond useful metadata. 
"""

__all__ = ["MyMultiClassAccuracy", "MyLoss"]

class MyMultiClassAccuracy(Metric):
    """ 
    My own multiclass accuracy metric store, for less headache. 
    """
    def __init__(self, labels: list[str], name: str = "acc", **kwargs):
        super().__init__(**kwargs)
        _ = [
            self.add_state(f"{lbl}", 
                default=[], 
                persistent=True,
                dist_reduce_fx="cat" # if there was dist metrics used this is used
            ) 
            for lbl in labels
        ]
        self._labels = labels
        self._name = name
        self.add_state("steps", [], persistent=True) # metadata state to track steps to epochs, not used in compute
        self.add_state("epochs", [], persistent=True)

    @property
    def name(self):
        return self._name
    
    def update(self, targets: torch.Tensor, preds: torch.Tensor, step: int, epoch: int) -> None:
        """ 
        TODO: Update for this metric from GCH inspired. 
        """
        unique_labels = list(targets.unique())
        all_labels = list(torch.as_tensor(range(0, len(self._labels))))
        for i in all_labels:
            if i in unique_labels:
                total = (targets == i).sum()
                correct = (preds[targets == i] == i).sum()
                avg = correct / total
                curr_list = getattr(self, self._labels[i])
                assert type(curr_list) is list, "Layer values must be a list for CH store"

                curr_list.append(avg)
                setattr(self, self._labels[i], curr_list)
            else:
                curr_list = getattr(self, self._labels[i])
                assert type(curr_list) is list, "Layer values must be a list for CH store"

                curr_list.append(torch.as_tensor(100))
                setattr(self, self._labels[i], curr_list)

        assert type(self.steps) is list, "Steps must be list of GC metric"
        self.steps.append(torch.tensor(step, dtype=torch.int32))
        
        assert type(self.epochs) is list, "Epochs must be list of GC metric"
        self.epochs.append(torch.tensor(epoch, dtype=torch.int32))

    def compute(self, mode: str = "all") -> dict[str, list[torch.Tensor]] | dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """ 
        Returns a dict of each lbl name and its paired list of tensor scalar values for "all"
        In other reduction modes we may get a single tensor summary value, or the value preceded by metadata, such as the step_idx. 
        """
        layer_dict = {lbl: getattr(self, lbl) for lbl in self._labels}

        if mode == "all":
            return layer_dict
        elif mode == "mean":
            mean_dict = {lbl: torch.as_tensor(getattr(self, lbl)).mean() for lbl in self._labels}
            return mean_dict
        elif mode == "last":
            last_dict = {lbl: getattr(self, lbl)[-1] for lbl in self._labels}
            return last_dict   
        elif mode == "max":
            max_dict = {lbl: (torch.as_tensor(getattr(self, lbl)).argmax(), torch.as_tensor(getattr(self, lbl)).max()) for lbl in self._labels}
            return max_dict        

        return layer_dict
    
class MyLoss(Metric):
    """
    My own loss metric store, for less headache. 
    """
    plot_lower_bound = -1
    plot_upper_bound = 1
    def __init__(self, name: str = "loss", **kwargs):
        super().__init__(**kwargs)
        self._name = name
        self.add_state("steps", [], persistent=True) # metadata state to track steps to epochs, not used in compute
        self.add_state("epochs", [], persistent=True)        
        self.add_state("loss", [], persistent=True)        

    def update(self, loss: torch.Tensor, step: int, epoch: int) -> None:
        """ 
        Update with loss, step, epoch into their respective lists. 
        """
        curr_list = getattr(self, "loss")
        assert type(curr_list) is list, "Layer values must be a list for CH store"

        curr_list.append(loss)
        setattr(self, "loss", curr_list)

        assert type(self.steps) is list, "Steps must be list of GC metric"
        self.steps.append(torch.tensor(step, dtype=torch.int32))
        
        assert type(self.epochs) is list, "Epochs must be list of GC metric"
        self.epochs.append(torch.tensor(epoch, dtype=torch.int32))

    def compute(self, mode: str = "all") -> Any:
        """ 
        Returns loss as "all" values or "mean" of values. 
        """
        # assert type(list(self.loss)) is list[torch.Tensor], f"Loss of type {type(self.loss)} is not list of tensor in metrics tracker"

        if mode == "all":
            loss = self.loss
        elif mode == "mean":
            loss = torch.as_tensor(self.loss)
            loss = loss.mean(dim=0)

        return dict(loss = self.loss) # fallback if modes not parseable

if __name__ == "__main__":
    pass