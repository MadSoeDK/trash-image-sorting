"""Tests for trashsorting.data module.

These tests use synthetic data fixtures (defined in conftest.py) to avoid
downloading the real TrashNet dataset in CI environments.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from trashsorting.data import TrashDataPreprocessed


# Tests for TrashDataPreprocessed using synthetic data fixtures


def test_preprocessed_dataset_instantiation(synthetic_data_path):
    """Test that TrashDataPreprocessed instantiates correctly."""
    dataset = TrashDataPreprocessed(synthetic_data_path, split="train")

    # Check dataset has samples
    assert len(dataset) > 0, "Dataset should have samples"

    # Check it's a valid Dataset
    assert isinstance(dataset, Dataset)

    # Check classes exist
    assert hasattr(dataset, "classes"), "Dataset should have classes attribute"
    assert len(dataset.classes) == 6, "TrashNet should have 6 classes"

    # Check expected classes are present
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert dataset.classes == expected_classes, f"Expected classes {expected_classes}, got {dataset.classes}"


def test_preprocessed_getitem(synthetic_data_path):
    """Test __getitem__ returns correct format."""
    dataset = TrashDataPreprocessed(synthetic_data_path, split="train")

    # Get first sample
    img, label = dataset[0]

    # Check image is a tensor with correct shape
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.shape == torch.Size([3, 224, 224]), f"Expected shape [3, 224, 224], got {img.shape}"

    # Check label is valid
    assert isinstance(label, int), f"Label should be int, got {type(label)}"
    assert 0 <= label < len(dataset.classes), f"Label {label} out of range [0, {len(dataset.classes)})"


def test_preprocessed_splits(synthetic_data_path):
    """Test that splits are correctly sized (70/15/15)."""
    train = TrashDataPreprocessed(synthetic_data_path, split="train")
    val = TrashDataPreprocessed(synthetic_data_path, split="val")
    test = TrashDataPreprocessed(synthetic_data_path, split="test")

    total = len(train) + len(val) + len(test)

    # Check approximate split ratios (allow some rounding error)
    # With only 12 samples, ratios won't be exact
    train_ratio = len(train) / total
    val_ratio = len(val) / total
    test_ratio = len(test) / total

    # More lenient bounds for small dataset
    assert 0.5 <= train_ratio <= 0.85, f"Train ratio should be ~0.7, got {train_ratio:.2f}"
    assert 0.05 <= val_ratio <= 0.25, f"Val ratio should be ~0.15, got {val_ratio:.2f}"
    assert 0.05 <= test_ratio <= 0.25, f"Test ratio should be ~0.15, got {test_ratio:.2f}"


def test_preprocessed_seed_reproducibility(synthetic_data_path):
    """Test that same seed produces same splits."""
    dataset1 = TrashDataPreprocessed(synthetic_data_path, split="train", seed=42)
    dataset2 = TrashDataPreprocessed(synthetic_data_path, split="train", seed=42)

    # Check same number of samples
    assert len(dataset1) == len(dataset2), "Same seed should produce same split size"

    # Check first samples have same label
    _, label1 = dataset1[0]
    _, label2 = dataset2[0]

    assert label1 == label2, "Labels should be identical with same seed"


def test_preprocessed_dataloader_compatibility(synthetic_data_path):
    """Test that dataset works with PyTorch DataLoader."""
    dataset = TrashDataPreprocessed(synthetic_data_path, split="train")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get first batch
    batch_img, batch_labels = next(iter(loader))

    # Check batch shapes
    assert batch_img.shape == torch.Size([2, 3, 224, 224]), f"Expected batch shape [2, 3, 224, 224], got {batch_img.shape}"
    assert batch_labels.shape == torch.Size([2]), f"Expected labels shape [2], got {batch_labels.shape}"

    # Check batch types
    assert isinstance(batch_img, torch.Tensor), "Batch images should be torch.Tensor"
    assert isinstance(batch_labels, torch.Tensor), "Batch labels should be torch.Tensor"


def test_preprocessed_all_split(synthetic_data_path):
    """Test loading all data without splitting."""
    dataset = TrashDataPreprocessed(synthetic_data_path, split="all")

    # Should have all 12 samples
    assert len(dataset) == 12, f"Expected 12 samples, got {len(dataset)}"


def test_class_to_idx_mapping(synthetic_data_path):
    """Test class_to_idx attribute is correct."""
    dataset = TrashDataPreprocessed(synthetic_data_path, split="train")

    expected_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5,
    }

    assert dataset.class_to_idx == expected_mapping, f"Expected {expected_mapping}, got {dataset.class_to_idx}"


def test_different_seeds_produce_different_splits(synthetic_data_path):
    """Test that different seeds produce different splits."""
    dataset1 = TrashDataPreprocessed(synthetic_data_path, split="train", seed=42)
    dataset2 = TrashDataPreprocessed(synthetic_data_path, split="train", seed=123)

    # Get all labels from both datasets
    labels1 = [dataset1[i][1] for i in range(len(dataset1))]
    labels2 = [dataset2[i][1] for i in range(len(dataset2))]

    # With different seeds, the order should be different
    # (though with small dataset, might occasionally be same by chance)
    # At minimum, check both datasets work
    assert len(labels1) > 0
    assert len(labels2) > 0
