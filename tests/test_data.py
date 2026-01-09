import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from trashsorting.data import TrashData, TrashDataPreprocessed


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = TrashData("data/raw")
    assert isinstance(dataset, Dataset)


def test_dataset_instantiation():
    """Test dataset instantiation and basic properties."""
    dataset = TrashData("data/raw", split="train")

    # Check dataset has samples
    assert len(dataset) > 0, "Dataset should have samples"

    # Check classes exist
    assert hasattr(dataset, "classes"), "Dataset should have classes attribute"
    assert len(dataset.classes) == 6, "TrashNet should have 6 classes"

    # Check expected classes are present
    expected_classes = ["cardboard", "glass",
                        "metal", "paper", "plastic", "trash"]
    assert dataset.classes == expected_classes, f"Expected classes {expected_classes}, got {dataset.classes}"


def test_getitem():
    """Test __getitem__ returns correct format."""
    dataset = TrashData("data/raw", split="train")

    # Get first sample
    img, label = dataset[0]

    # Check image is a tensor with correct shape
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.shape == torch.Size(
        [3, 224, 224]), f"Expected shape [3, 224, 224], got {img.shape}"

    # Check label is valid
    assert isinstance(label, int), f"Label should be int, got {type(label)}"
    assert 0 <= label < len(
        dataset.classes), f"Label {label} out of range [0, {len(dataset.classes)})"

    # Check we can map label to class name
    class_name = dataset.classes[label]
    assert class_name in dataset.classes, f"Class name {class_name} not in classes"


def test_dataloader_compatibility():
    """Test dataset works with PyTorch DataLoader."""
    dataset = TrashData("data/raw", split="train")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get first batch
    batch_img, batch_labels = next(iter(loader))

    # Check batch shapes
    assert batch_img.shape == torch.Size(
        [2, 3, 224, 224]), f"Expected batch shape [2, 3, 224, 224], got {batch_img.shape}"
    assert batch_labels.shape == torch.Size(
        [2]), f"Expected labels shape [2], got {batch_labels.shape}"

    # Check batch types
    assert isinstance(
        batch_img, torch.Tensor), "Batch images should be torch.Tensor"
    assert isinstance(
        batch_labels, torch.Tensor), "Batch labels should be torch.Tensor"


# Tests for TrashDataPreprocessed
# Note: These tests work with either preprocessed data (if available) or fallback to TrashData

def test_preprocessed_dataset_instantiation():
    """Test that TrashDataPreprocessed instantiates correctly."""
    # Use small fraction for faster testing
    dataset = TrashDataPreprocessed("data", split="train", fraction=0.1)

    # Check dataset has samples
    assert len(dataset) > 0, "Dataset should have samples"

    # Check classes exist
    assert hasattr(dataset, "classes"), "Dataset should have classes attribute"
    assert len(dataset.classes) == 6, "TrashNet should have 6 classes"

    # Check expected classes are present
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert dataset.classes == expected_classes, f"Expected classes {expected_classes}, got {dataset.classes}"


def test_preprocessed_getitem():
    """Test __getitem__ returns correct format."""
    dataset = TrashDataPreprocessed("data", split="train", fraction=0.1)

    # Get first sample
    img, label = dataset[0]

    # Check image is a tensor with correct shape
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.shape == torch.Size([3, 224, 224]), f"Expected shape [3, 224, 224], got {img.shape}"

    # Check label is valid
    assert isinstance(label, int), f"Label should be int, got {type(label)}"
    assert 0 <= label < len(dataset.classes), f"Label {label} out of range [0, {len(dataset.classes)})"


def test_preprocessed_splits():
    """Test that splits are correctly sized (70/15/15)."""
    # Use small fraction for faster testing
    train = TrashDataPreprocessed("data", split="train", fraction=0.1)
    val = TrashDataPreprocessed("data", split="val", fraction=0.1)
    test = TrashDataPreprocessed("data", split="test", fraction=0.1)

    total = len(train) + len(val) + len(test)

    # Check approximate split ratios (allow some rounding error)
    train_ratio = len(train) / total
    val_ratio = len(val) / total
    test_ratio = len(test) / total

    assert 0.68 <= train_ratio <= 0.72, f"Train ratio should be ~0.7, got {train_ratio:.2f}"
    assert 0.13 <= val_ratio <= 0.17, f"Val ratio should be ~0.15, got {val_ratio:.2f}"
    assert 0.13 <= test_ratio <= 0.17, f"Test ratio should be ~0.15, got {test_ratio:.2f}"


def test_preprocessed_seed_reproducibility():
    """Test that same seed produces same splits."""
    # Use small fraction for faster testing
    dataset1 = TrashDataPreprocessed("data", split="train", fraction=0.1, seed=42)
    dataset2 = TrashDataPreprocessed("data", split="train", fraction=0.1, seed=42)

    # Check same number of samples
    assert len(dataset1) == len(dataset2), "Same seed should produce same split size"

    # Check first samples have same label
    _, label1 = dataset1[0]
    _, label2 = dataset2[0]

    assert label1 == label2, "Labels should be identical with same seed"


def test_preprocessed_dataloader_compatibility():
    """Test that dataset works with PyTorch DataLoader."""
    dataset = TrashDataPreprocessed("data", split="train", fraction=0.1)

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
