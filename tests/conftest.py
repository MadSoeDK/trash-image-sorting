"""Pytest fixtures for trashsorting tests."""

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def synthetic_data_path(tmp_path_factory) -> Path:
    """Create a synthetic preprocessed dataset for testing.

    Creates a tiny preprocessed dataset with 12 random images (2 per class)
    in a completely isolated temporary directory. This fixture is safe and
    will never touch real data in the project's data/ directory.

    Returns:
        Path to the isolated temporary data directory containing synthetic data.
    """
    # Use pytest's tmp_path_factory for completely isolated temp directory
    data_path = tmp_path_factory.mktemp("synthetic_test_data")
    processed_path = data_path / "processed"
    processed_path.mkdir(parents=True)

    # TrashNet classes
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Create synthetic images (2 per class = 12 total)
    num_per_class = 2
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        for _ in range(num_per_class):
            # Create a random tensor image (3, 224, 224)
            # Use different color tints per class for visual distinction
            img = torch.rand(3, 224, 224)
            # Add class-specific color bias
            img[class_idx % 3] += 0.3
            img = img.clamp(0, 1)

            images.append(img)
            labels.append(class_idx)

    # Stack into tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create metadata matching the real preprocessed format
    metadata = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_samples": len(images),
        "preprocessing_date": "2024-01-01T00:00:00",
        "transform_applied": "Resize(224,224), ToTensor, ImageNet Normalize",
        "seed": 42,
        "fraction": 1.0,
        "pytorch_version": torch.__version__,
        "torchvision_version": "0.0.0",
    }

    # Save preprocessed file in the isolated temp directory
    output_file = processed_path / "trashnet.pt"
    torch.save(
        {
            "images": images_tensor,
            "labels": labels_tensor,
            "metadata": metadata,
        },
        output_file,
    )

    return data_path
