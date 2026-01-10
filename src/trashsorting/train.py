from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel
import typer
import logging

logger = logging.getLogger(__name__)

def train(
    fraction: float = 1.0,
    batch_size: int = 32,
    max_epochs: int = 10
):
    model = TrashModel()

    # Configure checkpoint callback to save best and last models
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(TrashDataPreprocessed("data", split="train", fraction=fraction), batch_size=batch_size, shuffle=True),
        val_dataloaders=DataLoader(TrashDataPreprocessed("data", split="val", fraction=fraction), batch_size=batch_size, shuffle=False)
    )

    logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    typer.run(train)
