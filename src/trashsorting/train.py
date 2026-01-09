from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel
import typer

def train(
        fraction: float = 1.0,
        batch_size: int = 32,
        max_epochs: int = 10
):
    model = TrashModel()
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model,
                train_dataloaders=DataLoader(TrashDataPreprocessed("data", split="train", fraction=fraction), batch_size=batch_size, shuffle=True),
                val_dataloaders=DataLoader(TrashDataPreprocessed("data", split="val", fraction=fraction), batch_size=batch_size, shuffle=False))


if __name__ == "__main__":
    typer.run(train)
