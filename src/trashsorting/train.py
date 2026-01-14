from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path

import torch
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
        train_dataloaders=DataLoader(TrashDataPreprocessed("data", split="train", fraction=fraction), batch_size=batch_size, shuffle=True),
        val_dataloaders=DataLoader(TrashDataPreprocessed("data", split="val", fraction=fraction), batch_size=batch_size, shuffle=False)
    )

    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")

    # Save the final model to models/model.pth for DVC tracking
    model_output_path = Path("models")
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Save the best model's state dict
    if checkpoint_callback.best_model_path:
        best_model = TrashModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        torch.save(best_model.state_dict(), model_output_path / "model.pth")
    else:
        torch.save(model.state_dict(), model_output_path / "model.pth")

    print(f"Training complete! Model saved to {model_output_path / 'model.pth'}")


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
