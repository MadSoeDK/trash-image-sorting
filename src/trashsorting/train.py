from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from trashsorting.data import TrashData
from trashsorting.model import TrashModel

def train_dataloader(self):
    return

def train():
    model = TrashModel()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model,
                train_dataloaders=DataLoader(TrashData("data/raw", split="train"), batch_size=16, shuffle=True),
                val_dataloaders=DataLoader(TrashData("data/raw", split="val"), batch_size=16, shuffle=False))


if __name__ == "__main__":
    train()
