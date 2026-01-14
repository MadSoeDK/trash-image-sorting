from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import yaml
from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel
import logging
from dotenv import load_dotenv
import os
load_dotenv()

logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def train(
    fraction: float = 1.0,
    batch_size: int = 32,
    max_epochs: int = 10,
    use_wandb_logger: bool = True,
    num_workers: int = 4
):
    model = TrashModel()

    # Configure checkpoint callback to save best model with fixed name for DVC
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
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

    logger.info(f"Training complete! Model saved to {checkpoint_callback.best_model_path}")


def main():
    """Entry point that loads params from YAML."""
    params = load_params()
    train(
        fraction=params["data"]["fraction"],
        batch_size=params["train"]["batch_size"],
        max_epochs=params["train"]["max_epochs"],
        use_wandb_logger=params["train"]["use_wandb_logger"],
        num_workers=params["train"]["num_workers"],
    )


if __name__ == "__main__":
    main()
