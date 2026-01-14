from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import typer
import logging
import json
import yaml
import torch
from pathlib import Path

from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel

logger = logging.getLogger(__name__)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def evaluate(
    checkpoint_path: str = None,
    data_path: str = None,
    batch_size: int = None,
    num_workers: int = None,
    fraction: float = None,
    metrics_output: str = None,
) -> dict[str, float]:
    """Evaluate a trained model on the test dataset.

    Args:
        checkpoint_path: Path to the model checkpoint file (.pth). If None, reads from params.
        data_path: Root directory for data. If None, reads from params.
        batch_size: Batch size for evaluation. If None, reads from params.
        num_workers: Number of workers for data loading. If None, reads from params.
        fraction: Fraction of test data to use. If None, reads from params.
        metrics_output: Path to save metrics JSON file. If None, reads from params.

    Returns:
        Dictionary containing evaluation metrics
            test_loss: float
            test_accuracy: float
    """
    # Load parameters from params.yaml
    params = load_params()
    data_params = params["data"]
    eval_params = params.get("evaluate", {})

    # Use provided arguments or fall back to params
    checkpoint_path = checkpoint_path or eval_params.get("checkpoint_path", "models/model.pth")
    data_path = data_path or eval_params.get("data_path", "data")
    batch_size = batch_size or eval_params.get("batch_size", 32)
    num_workers = num_workers or eval_params.get("num_workers", 4)
    fraction = fraction if fraction is not None else data_params.get("fraction", 1.0)
    metrics_output = metrics_output or eval_params.get("metrics_output", "reports/eval_metrics.json")

    print("=" * 60)
    print("Starting evaluation with parameters:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data path: {data_path}")
    print(f"  Data fraction: {fraction}")
    print(f"  Batch size: {batch_size}")
    print(f"  Metrics output: {metrics_output}")
    print("=" * 60)

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Load preprocessed data to get metadata
    processed_path = Path(data_path) / "processed"
    metadata = torch.load(processed_path / "all_metadata.pt", weights_only=False)
    num_classes = len(metadata["classes"])

    # Load model from checkpoint - handle both .pth and .ckpt files
    if checkpoint_path.endswith('.ckpt'):
        model = TrashModel.load_from_checkpoint(checkpoint_path)
    else:
        # Load from state dict (.pth file)
        from trashsorting.train import load_params as load_train_params
        train_params = load_train_params()["train"]
        model = TrashModel(
            model_name=train_params.get("model_name", "resnet18"),
            num_classes=num_classes,
            lr=train_params.get("learning_rate", 0.001)
        )
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    model.eval()

    # Load test dataset
    logger.info(f"Loading test dataset from: {data_path}")
    test_dataset = TrashDataPreprocessed(data_path, split="test", fraction=fraction)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    logger.info(f"Evaluating on {len(test_dataset)} test samples...")
    print(f"\nEvaluating on {len(test_dataset)} test samples...")

    # Create trainer for evaluation
    trainer = Trainer(logger=False, enable_checkpointing=False)
    results = trainer.test(model, dataloaders=test_loader)

    # Extract metrics
    metrics = dict(results[0]) if results else {}

    logger.info(f"Results: {metrics}")
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save metrics to JSON file for DVC tracking
    output_path = Path(metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_path}")
    logger.info(f"Metrics saved to {output_path}")

    return metrics


if __name__ == "__main__":
    typer.run(evaluate)
