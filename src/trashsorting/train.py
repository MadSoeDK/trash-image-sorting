from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel
import typer
import logging
from dotenv import load_dotenv
import os
load_dotenv()

logger = logging.getLogger(__name__)

def train(
    fraction: float = 1.0,
    batch_size: int = 32,
    max_epochs: int = 10,
    use_wandb_logger: bool = True,
    num_workers: int = 4
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

    # initialise the wandb logger
    wandb_logger = None
    if use_wandb_logger:
        wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT"))
        wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )

    trainer.fit(
        model,
        train_dataloaders=DataLoader(TrashDataPreprocessed("data", split="train", fraction=fraction), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_dataloaders=DataLoader(TrashDataPreprocessed("data", split="val", fraction=fraction), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

    logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    typer.run(train)
