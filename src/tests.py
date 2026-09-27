from torchmetrics import Metric
import torchmetrics
import torch

num_updates = 3
metric = torchmetrics.Accuracy(task="binary")
for _ in range(num_updates):
    metric.update(torch.rand(10,), torch.randint(2, (10,)))
fig, ax = metric.plot()