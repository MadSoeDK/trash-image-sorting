from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path

import torch
import yaml
from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel
import typer
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


def train():
    """Train the trash classification model using DVC-tracked parameters and data."""
    # Load parameters
    params = load_params()
    data_params = params["data"]
    train_params = params["train"]

    print("=" * 60)
    print("Starting training with parameters:")
    print(f"  Data fraction: {data_params['fraction']}")
    print(f"  Model: {train_params['model_name']}")
    print(f"  Learning rate: {train_params['learning_rate']}")
    print(f"  Batch size: {train_params['batch_size']}")
    print(f"  Epochs: {train_params['epochs']}")
    print("=" * 60)

    fraction: float = data_params.get("fraction", 1.0)
    seed: int = data_params.get("seed", 42)
    batch_size: int = train_params.get("batch_size", 32)
    max_epochs: int = train_params.get("epochs", 10)
    learning_rate: float = train_params.get("learning_rate", 0.001)
    model_name: str = train_params.get("model_name", "resnet18")

    # Load preprocessed data to get metadata
    processed_path = Path("data/processed")
    print(f"\nLoading preprocessed data from {processed_path}...")

    metadata = torch.load(processed_path / "all_metadata.pt", weights_only=False)
    num_classes = len(metadata["classes"])
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {metadata['classes']}")

    # Create datasets using TrashDataPreprocessed
    train_dataset = TrashDataPreprocessed("data", split="train", fraction=fraction, seed=seed)
    val_dataset = TrashDataPreprocessed("data", split="val", fraction=fraction, seed=seed)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model with parameters from config
    model = TrashModel(
        model_name=model_name,
        num_classes=num_classes,
        lr=learning_rate
    )

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


if __name__ == "__main__":
    typer.run(train)
