from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import logging
import json
import yaml
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
    checkpoint_path: str,
    data_path: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    fraction: float = 1.0,
    output: str = "models/metrics.json"
) -> dict[str, float]:
    """Evaluate a trained model on the test dataset.

    Args:
        checkpoint_path: Path to the model checkpoint file (.pth). If None, reads from params.
        data_path: Root directory for data. If None, reads from params.
        batch_size: Batch size for evaluation. If None, reads from params.
        num_workers: Number of workers for data loading. If None, reads from params.
        fraction: Fraction of test data to use. If None, reads from params.
        output: Path to save metrics JSON file. If None, reads from params.

    Returns:
        Dictionary containing evaluation metrics
            test_loss: float
            test_accuracy: float
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Load preprocessed data to get metadata
    model = TrashModel.load_from_checkpoint(checkpoint_path)
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
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")

    return metrics


def main():
    # Load parameters from params.yaml
    params = load_params()
    evaluate(
        checkpoint_path=params["evaluate"]["checkpoint_path"],
        data_path=params["evaluate"]["data_path"],
        batch_size=params["evaluate"]["batch_size"],
        num_workers=params["evaluate"]["num_workers"],
        fraction=params["data"]["fraction"],
        output=params["evaluate"]["output"]
    )

if __name__ == "__main__":
    main()
