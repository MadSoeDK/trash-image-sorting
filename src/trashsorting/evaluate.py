from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import typer
import logging

from trashsorting.data import TrashDataPreprocessed
from trashsorting.model import TrashModel

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    data_path: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    fraction: float = 1.0,
) -> dict[str, float]:
    """Evaluate a trained model on the test dataset.

    Args:
        checkpoint_path: Path to the model checkpoint file (.ckpt)
        data_path: Root directory for data (default: "data")
        batch_size: Batch size for evaluation (default: 32)
        num_workers: Number of workers for data loading (default: 4)
        fraction: Fraction of test data to use (default: 1.0)

    Returns:
        Dictionary containing evaluation metrics
            test_loss: float
            test_accuracy: float
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Load model from checkpoint
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

    # Create trainer for evaluation
    trainer = Trainer(logger=False, enable_checkpointing=False)
    results = trainer.test(model, dataloaders=test_loader)

    logger.info(f"Results: {results}")
    print(results)

    return dict(results[0]) if results else {}


if __name__ == "__main__":
    typer.run(evaluate)
