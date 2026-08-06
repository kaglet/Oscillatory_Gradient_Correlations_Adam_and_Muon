import torchvision

def mnist_normalization():
    return torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )