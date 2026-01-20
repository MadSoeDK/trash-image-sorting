"""Dataset statistics script for continuous machine learning workflows.

This script generates statistics and visualizations about the trash dataset
to be used in automated data quality checks when data changes.
"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import torch
import typer

from trashsorting.data import TrashDataPreprocessed

logger = logging.getLogger(__name__)
app = typer.Typer()


def generate_dataset_statistics(data_path: str = "data") -> None:
    """Generate and report dataset statistics.

    Args:
        data_path: Path to the data directory
    """
    # Load datasets
    train_dataset = TrashDataPreprocessed(data_path, split="train")
    val_dataset = TrashDataPreprocessed(data_path, split="val")
    test_dataset = TrashDataPreprocessed(data_path, split="test")

    # Get basic statistics
    print("# Dataset Statistics Report\n")
    print("## Dataset Sizes")
    print(f"- **Training samples**: {len(train_dataset)}")
    print(f"- **Validation samples**: {len(val_dataset)}")
    print(f"- **Test samples**: {len(test_dataset)}")
    print(f"- **Total samples**: {len(train_dataset) + len(val_dataset) + len(test_dataset)}\n")

    # Get class information
    print("## Classes")
    print(f"- **Number of classes**: {len(train_dataset.classes)}")
    print(f"- **Class names**: {', '.join(train_dataset.classes)}\n")

    # Calculate class distribution
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    print("## Class Distribution\n")
    print("### Training Set")
    for class_idx, class_name in enumerate(train_dataset.classes):
        count = train_counts.get(class_idx, 0)
        percentage = (count / len(train_dataset)) * 100 if len(train_dataset) > 0 else 0
        print(f"- **{class_name}**: {count} samples ({percentage:.1f}%)")

    print("\n### Validation Set")
    for class_idx, class_name in enumerate(val_dataset.classes):
        count = val_counts.get(class_idx, 0)
        percentage = (count / len(val_dataset)) * 100 if len(val_dataset) > 0 else 0
        print(f"- **{class_name}**: {count} samples ({percentage:.1f}%)")

    print("\n### Test Set")
    for class_idx, class_name in enumerate(test_dataset.classes):
        count = test_counts.get(class_idx, 0)
        percentage = (count / len(test_dataset)) * 100 if len(test_dataset) > 0 else 0
        print(f"- **{class_name}**: {count} samples ({percentage:.1f}%)")

    # Generate visualizations
    generate_sample_images(train_dataset)
    generate_distribution_plots(train_dataset, val_dataset, test_dataset,
                                train_counts, val_counts, test_counts)

    print("\n## Visualizations Generated")
    print("- Sample images: `trash_sample_images.png`")
    print("- Training distribution: `train_label_distribution.png`")
    print("- Validation distribution: `val_label_distribution.png`")
    print("- Test distribution: `test_label_distribution.png`")


def generate_sample_images(dataset: TrashDataPreprocessed, num_samples: int = 9) -> None:
    """Generate a grid of sample images from the dataset.

    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Sample Images from Training Dataset", fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx < num_samples and idx < len(dataset):
            image, label = dataset[idx]
            # Denormalize image for visualization
            # Assuming ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)

            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(f"{dataset.classes[label]}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("trash_sample_images.png", dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("Sample images saved to trash_sample_images.png")


def generate_distribution_plots(train_dataset, val_dataset, test_dataset,
                                train_counts, val_counts, test_counts) -> None:
    """Generate bar plots showing class distributions.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        train_counts: Counter object with training label counts
        val_counts: Counter object with validation label counts
        test_counts: Counter object with test label counts
    """
    # Training distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = train_dataset.classes
    counts = [train_counts.get(i, 0) for i in range(len(class_names))]
    ax.bar(class_names, counts, color='steelblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Training Set - Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("train_label_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()

    # Validation distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = [val_counts.get(i, 0) for i in range(len(class_names))]
    ax.bar(class_names, counts, color='darkgreen')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Validation Set - Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("val_label_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()

    # Test distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = [test_counts.get(i, 0) for i in range(len(class_names))]
    ax.bar(class_names, counts, color='firebrick')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Test Set - Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("test_label_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()

    logger.info("Distribution plots saved")


@app.command()
def main(data_path: str = "data") -> None:
    """Main entry point for the dataset statistics script.

    Args:
        data_path: Path to the data directory (default: 'data')
    """
    logging.basicConfig(level=logging.INFO)
    generate_dataset_statistics(data_path)


if __name__ == "__main__":
    app()

