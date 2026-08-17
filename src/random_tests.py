import multiprocessing

from torchvision.models import resnet18

# names = set()
# for n,p in resnet18().named_parameters():
#     names.add(n)
#     print(n)

print(multiprocessing.cpu_count())