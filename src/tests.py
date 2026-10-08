from collections import OrderedDict
import logging
from pathlib import Path

import torchmetrics
from torchmetrics import Accuracy, Metric
import torch 
import matplotlib.pyplot as plt

from torchvision.models import resnet18

from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import MaxMetric, CatMetric, Metric, MetricCollection

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy

def binary_plot(num_updates=3, title="This is a nice plot", path="tests/my_binary_plot.png"):
    metric = torchmetrics.Accuracy(task="binary")
    for _ in range(num_updates):
        metric.update(torch.rand(10,), torch.randint(2, (10,)))
    fig, ax = metric.plot()

    # ax.set_fontsize(fs=20)
    fig.suptitle(title)
    fig.savefig(path)

def multi_cls_plot(num_updates=3, title="This is a nice plot", path="tests/my_multicls_plot.png"):
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None)
    for _ in range(num_updates):
        metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
        fig, ax = metric.plot()

    fig.suptitle(title) # type: ignore
    fig.savefig(path) # type: ignore

def non_scalar_a(num_updates=3, title="This is a nice plot", path="tests/my_nonscalar-a_plot.png"):
    metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)
    for _ in range(num_updates):
        metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
    fig, ax = metric.plot()
    fig.suptitle(title) # type: ignore
    fig.savefig(path) # type: ignore

def multi_step(epochs = 3, num_updates=3, title="This is a nice plot", path="tests/my_multistep_plot.png"):
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None)
    values = [ ]
    for step in range(epochs):
        for _ in range(num_updates):
            metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
        values.append(metric.compute())  # save value
        metric.reset()
    fig, ax = metric.plot(values)
    fig.suptitle(title) 
    fig.savefig(path) 

def metric_collection(epochs = 3, num_updates=3, title="This is a nice plot", path="tests/my_metric-collection_plot.png"):
    collection = torchmetrics.MetricCollection(
        torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None),
        torchmetrics.Recall(task="multiclass", num_classes=3, average=None),
        torchmetrics.Precision(task="multiclass", num_classes=3, average=None),
    )
    fig, ax = plt.subplots(nrows=1, ncols=3)
    values = [ ]
    for _ in range(epochs):
        for _ in range(num_updates):
            collection.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
        values.append(collection.compute())
        collection.reset()
    collection.plot(val=list(values[0]["MulticlassAccuracy"]), ax=list(ax))
    fig.suptitle(title) 
    fig.savefig(path) 

# binary_plot(3)
# multi_cls_plot(3)
# non_scalar_a(3)
# multi_step()
# metric_collection()
def analyse_metrics():
    count_each = dict(gc=0, rgc=0, train_acc=0, val_acc=0, train_loss=0, val_loss=0)

    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    print(outputs_dir.exists())
    all_metrics = sorted(outputs_dir.rglob("*.pt"))
    cif10_labels = ["airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"]
    for idx, path in enumerate(all_metrics):
        # Step numbers are correct for acc vs GC (1 behind)
        metric = torch.load(path)
        # assert type(metric) is OrderedDict | dict, f"Metric of {type(metric)} is not OrderedDict | dict"
        result = None
        path = str(path.stem).lower()
        if not "gc" in path:
            if "val_acc" in path and count_each["val_acc"] < 1:
                base_acc = MyMultiClassAccuracy(cif10_labels)
                base_acc.load_state_dict(metric)
                result = base_acc.compute()
                count_each["val_acc"] += 1
            elif "val_loss" in path and count_each["val_loss"] < 1:
                base_loss = MyLoss()
                base_loss.load_state_dict(metric)
                result = base_loss.compute()
                count_each["val_loss"] += 1
            if "train_acc" in path and count_each["train_acc"] < 1:
                base_acc = MyMultiClassAccuracy(cif10_labels)
                base_acc.load_state_dict(metric)
                result = base_acc.compute()
                count_each["train_acc"] += 1
            elif "train_loss" in path and count_each["train_loss"] < 1:
                base_loss = MyLoss()
                base_loss.load_state_dict(metric)
                result = base_loss.compute()
                count_each["train_loss"] += 1
        else: 
            header = list(metric.keys())

            if not "rgc" in path and count_each["gc"] < 1:
                metric_gch = GCH_Metric(header=header, name="")
                metric_gch.load_state_dict(metric)
                result = metric_gch.compute()
                count_each["gc"] += 1
            elif "rgc" in path and count_each["rgc"] < 1: 
                metric_gch = RGCH_Metric(header=header, name="")
                metric_gch.load_state_dict(metric)
                result = metric_gch.compute()         
                count_each["rgc"] += 1

        
        logging.basicConfig(filename=f"{Path(__file__).parent.resolve()}/metric_tests_{idx}.log", level=logging.DEBUG)

        logging.debug("Metric unknown: {}".format(result))

analyse_metrics()

# def repeat_join_str(val: str | int | float, repeat_count:int = 5) -> str:
#     val = str(val) if type(val) is int else val
#     string = f"{val},"
#     string *= repeat_count
#     string = string[:-1]
#     return string   
      

# print(repeat_join_str(1e-4, 5))
