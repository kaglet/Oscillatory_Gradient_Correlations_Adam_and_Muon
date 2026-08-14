from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, LightningDataModule, LightningModule

def train_and_val(
    lit_module: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
):
    print(type(trainer), type(datamodule), type(lit_module))
    print("we are in")
    trainer.fit(model=lit_module, datamodule=datamodule)
    # TODO: When you run test evaluate best AND last for Muon to see if degradation is a notably suffered effect
    trainer.test(model=lit_module, datamodule=datamodule)
    
    return 1