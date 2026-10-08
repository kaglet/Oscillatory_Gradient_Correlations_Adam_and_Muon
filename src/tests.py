from collections import OrderedDict
import logging
import os
from pathlib import Path
from typing import Any

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


# ----------------------------------------- Metric logging below

def store_metrics_per_optim(outputs_dir: Path, count_each: dict[str, int]):
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
    all_metrics = sorted(outputs_dir.rglob("*.pt"))

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
                logging.debug("Metric val_acc: {}\n".format(result))
            elif "val_loss" in path and count_each["val_loss"] < 1:
                base_loss = MyLoss()
                base_loss.load_state_dict(metric)
                result = base_loss.compute()
                count_each["val_loss"] += 1
                logging.debug("Metric val_loss: {}\n".format(result))
            if "train_acc" in path and count_each["train_acc"] < 1:
                base_acc = MyMultiClassAccuracy(cif10_labels)
                base_acc.load_state_dict(metric)
                result = base_acc.compute()
                count_each["train_acc"] += 1
                logging.debug("Metric train_acc: {}\n".format(result))
            elif "train_loss" in path and count_each["train_loss"] < 1:
                base_loss = MyLoss()
                base_loss.load_state_dict(metric)
                result = base_loss.compute()
                count_each["train_loss"] += 1
                logging.debug("Metric train_loss: {}\n".format(result))
        else: 
            header = list(metric.keys())

            if not "rgc" in path and count_each["gc"] < 1:
                metric_gch = GCH_Metric(header=header, name="")
                metric_gch.load_state_dict(metric)
                result = metric_gch.compute()
                count_each["gc"] += 1
                logging.debug("Metric gc: {}\n".format(result))
            elif "rgc" in path and count_each["rgc"] < 1: 
                metric_gch = RGCH_Metric(header=header, name="")
                metric_gch.load_state_dict(metric)
                result = metric_gch.compute()         
                count_each["rgc"] += 1
                logging.debug("Metric rgc: {}\n".format(result))

def analyse_metrics(outputs_dir: Path):
    count_each = dict(gc=0, rgc=0, train_acc=0, val_acc=0, train_loss=0, val_loss=0)
    muon_outputs_dir = outputs_dir/ "muon_class_imbalance_ratios"

    logging.debug("----- STORING MUON METRICS -----")
    store_metrics_per_optim(muon_outputs_dir, count_each)

    count_each = dict(gc=0, rgc=0, train_acc=0, val_acc=0, train_loss=0, val_loss=0)
    adam_outputs_dir = outputs_dir/ "adam_class_imbalance_ratios"

    logging.debug("----- STORING ADAM METRICS -----")
    store_metrics_per_optim(adam_outputs_dir, count_each)

def delete_test_log_file(log_file: Path):
    os.remove(log_file)

def log_count_labels(desired_paths: list[Any], actual_paths: list[Any], stage:str):
    run_idx = 0
    for desired, actual in zip(desired_paths,actual_paths):
        actual_counts = torch.load(actual)
        desired_counts = torch.load(desired)

        logging.debug("Stage - {} :: Actual :: Run {} :: class counts: {}\n".format(stage, run_idx, actual_counts))
        logging.debug("Stage - {} :: Desired :: Run {} :: class counts: {}\n".format(stage, run_idx, desired_counts))

        run_idx += 1

def analyse_count_labels(outputs_subdir: Path):
    unused_paths = sorted(outputs_subdir.rglob("unused_class_counts.pt"))
    train_used_paths = sorted(outputs_subdir.rglob("train_actual_class_counts.pt"))
    val_used_paths = sorted(outputs_subdir.rglob("val_actual_class_counts.pt"))
    train_desired_paths = sorted(outputs_subdir.rglob("train_desired_class_counts.pt"))
    val_desired_paths = sorted(outputs_subdir.rglob("val_desired_class_counts.pt"))

    log_count_labels(desired_paths= train_desired_paths, actual_paths=train_used_paths, stage= "train")
    log_count_labels(desired_paths= val_desired_paths, actual_paths=val_used_paths, stage= "val")

test_log_file = f"{Path(__file__).parent.resolve()}/metric_tests.log"
delete_test_log_file(Path(test_log_file))

logging.basicConfig(filename=test_log_file, level=logging.DEBUG)

outputs_dir = Path(__file__).resolve().parent.parent / "outputs"

analyse_metrics(outputs_dir)

muon_outputs_dir = outputs_dir/ "muon_class_imbalance_ratios"
logging.debug("----- STORING MUON COUNT LABELS -----\n")
analyse_count_labels(muon_outputs_dir)

adam_outputs_dir = outputs_dir/ "adam_class_imbalance_ratios"
logging.debug("----- STORING ADAM COUNT LABELS -----\n")
analyse_count_labels(adam_outputs_dir)

# def repeat_join_str(val: str | int | float, repeat_count:int = 5) -> str:
#     val = str(val) if type(val) is int else val
#     string = f"{val},"
#     string *= repeat_count
#     string = string[:-1]
#     return string   
      

# print(repeat_join_str(1e-4, 5))
