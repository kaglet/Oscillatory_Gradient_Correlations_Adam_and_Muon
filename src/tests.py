import torchmetrics
from torchmetrics import Metric
import torch 
import matplotlib.pyplot as plt

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

binary_plot(3)
multi_cls_plot(3)
non_scalar_a(3)
multi_step()
metric_collection()