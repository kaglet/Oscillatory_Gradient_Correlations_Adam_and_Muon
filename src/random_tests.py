from hydra_zen import builds
from lightning.pytorch import Trainer
import torch

trainer_cfg = builds(Trainer)
print(trainer_cfg)

class Class1:
    def m(self):
        print("In Class1")

class Class2(Class1):
    def m(self):
        print("In Class2")
        super().m()

class Class3(Class1):
    def m(self):
        print("In Class3")
        super().m()

class Class4(Class2, Class3):
    def m(self):
        print("In Class4")   
        super().m()
     
print(torch.cuda.is_available())