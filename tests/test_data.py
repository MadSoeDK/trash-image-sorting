import torch
from torch.utils.data import DataLoader, Dataset

from trashsorting.data import TrashData


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
