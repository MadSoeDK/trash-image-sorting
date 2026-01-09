from pathlib import Path

import torch
import yaml
from torch.utils.data import TensorDataset


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

    # Load preprocessed data
    processed_path = Path("data/processed")
    print(f"\nLoading preprocessed data from {processed_path}...")

    images = torch.load(processed_path / "all_images.pt")
    labels = torch.load(processed_path / "all_labels.pt")
    metadata = torch.load(processed_path / "all_metadata.pt")

    num_classes = len(metadata["classes"])
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {metadata['classes']}")

    # Create dataset and split into train/val/test (70/15/15)
    dataset = TensorDataset(images, labels)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    #TODO: Implement the actual training loop here


    #Generate dummy training output (- models/model.pth)
    model_output_path = Path("models")
    model_output_path.mkdir(parents=True, exist_ok=True)
    dummy_model = torch.nn.Linear(10, num_classes)  # Dummy model
    torch.save(dummy_model.state_dict(), model_output_path / "model.pth")
    print(f"\nTraining complete! Model saved to {model_output_path / 'model.pth'}")



if __name__ == "__main__":
    train()
