from pathlib import Path
from typing import Literal, Tuple, Optional, List
from datetime import datetime
import logging

import torch
import torchvision
import typer
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from typing import Any, cast

from trashsorting.utils.transform import image_transform

logger = logging.getLogger(__name__)


class TrashData(Dataset):
    """PyTorch Dataset for TrashNet image classification.

    This dataset downloads and wraps the TrashNet dataset from HuggingFace
    (garythung/trashnet) which contains images of trash categorized into
    6 classes: cardboard, glass, metal, paper, plastic, and trash.

    Args:
        data_path: Root directory for caching the dataset. Raw data will be
            stored in subdirectories following the Cookiecutter Data Science
            structure.
        split: Dataset split to load. One of "train", "test", "val", or "all".
            Default is "train".
        transform: Optional torchvision transform to apply to images.
            If None, uses default transforms (resize to 224x224 and normalize).

    Attributes:
        data_path: Root directory path for data storage.
        split: The dataset split being used.
        dataset: The loaded HuggingFace dataset.
        transform: Image transforms applied during __getitem__.
        classes: List of class names.
        class_to_idx: Mapping from class names to indices.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: Literal["train", "test", "val", "all"] = "train",
        transform: transforms.Compose | None = None,
        fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize the TrashNet dataset.

        Downloads the dataset from HuggingFace if not already cached.

        Args:
            data_path: Root directory for caching the dataset.
            split: Dataset split to load ("train", "test", "val", or "all").
            transform: Optional custom transforms. If None, uses default.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Download/load full dataset from HuggingFace (only train split exists)
        logger.info(f"Loading TrashNet dataset (fraction: {fraction})...")
        full_dataset = load_dataset(
            "garythung/trashnet",
            split="train",
            cache_dir=str(self.data_path),
        )

        # Shuffle the dataset to ensure uniform sampling across classes
        full_dataset = full_dataset.shuffle(seed=seed)

        # Apply fraction after shuffling
        if fraction < 1.0:
            subset_size = int(len(full_dataset) * fraction)
            full_dataset = full_dataset.select(range(subset_size))

        # Extract class information
        self.classes = full_dataset.features["label"].names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Split into train/val/test (70/15/15) using compute_splits
        if split == "all":
            self.dataset = full_dataset
        else:
            total_size = len(full_dataset)
            split_indices, train_size, val_size, test_size = compute_splits(
                total_size, split, seed
            )

            # Create reproducible splits
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset, test_dataset = random_split(
                cast(Dataset[Any], full_dataset), [train_size, val_size, test_size], generator=generator
            )

            # Select the requested split
            if split == "train":
                self.dataset = train_dataset
            elif split == "val":
                self.dataset = val_dataset
            elif split == "test":
                self.dataset = test_dataset

        # Set up transforms
        if transform is None:
            # Default transforms: resize, convert to tensor, and normalize
            self.transform = image_transform()
        else:
            self.transform = transform

        logger.info(f"Dataset loaded: {len(self)} samples across {len(self.classes)} classes")

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (image_tensor, label) where image_tensor is a
            torch.Tensor of shape (C, H, W) and label is an integer
            class index.
        """
        sample = self.dataset[index]
        image = sample["image"]

        # Convert to RGB if necessary (some images might be grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)
        label = sample["label"]

        return image_tensor, label

    def preprocess(self, output_folder: str | Path, fraction: float = 1.0, seed: int = 42) -> None:
        """Preprocess the raw data and save it to a single file.

        This method iterates through the entire dataset (regardless of split),
        applies transforms, and saves all processed images and labels to a
        single .pt file for faster loading in subsequent training runs.

        Args:
            output_folder: Directory where preprocessed data will be saved.
                Typically 'data/processed' following Cookiecutter structure.
            fraction: Fraction of data to preprocess (0.0-1.0). Default 1.0.
            seed: Random seed used for shuffling. Default 42.
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a dataset with all data for preprocessing
        logger.info(f"Creating full dataset for preprocessing (fraction: {fraction})...")
        full_dataset = TrashData(
            self.data_path, split="all", transform=self.transform,
            fraction=fraction, seed=seed
        )

        logger.info(f"Preprocessing {len(full_dataset)} samples to {output_path}/trashnet.pt...")

        # Process all samples
        images = []
        labels = []

        for idx in range(len(full_dataset)):
            image_tensor, label = full_dataset[idx]
            images.append(image_tensor)
            labels.append(label)

            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(full_dataset)} samples...")

        # Stack tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        # Prepare metadata
        metadata = {
            "classes": full_dataset.classes,
            "class_to_idx": full_dataset.class_to_idx,
            "num_samples": len(full_dataset),
            "preprocessing_date": datetime.now().isoformat(),
            "transform_applied": "Resize(224,224), ToTensor, ImageNet Normalize",
            "seed": seed,
            "fraction": fraction,
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        }

        # Save everything in a single file
        save_data = {
            "images": images_tensor,
            "labels": labels_tensor,
            "metadata": metadata,
        }

        output_file = output_path / "trashnet.pt"
        torch.save(save_data, output_file)

        logger.info(f"Preprocessing complete! Saved {len(full_dataset)} samples to {output_file}")
        logger.info(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")


class TrashDataPreprocessed(Dataset):
    """PyTorch Dataset for preprocessed TrashNet data.

    Loads preprocessed data from data/processed/trashnet.pt for fast training.
    If the preprocessed file doesn't exist, falls back to using TrashData.

    This dataset provides 10-20x faster loading compared to TrashData by
    loading preprocessed tensors directly from disk rather than processing
    images on-the-fly.

    Args:
        data_path: Root directory for data. Preprocessed file should be at
            data_path/processed/trashnet.pt, raw data at data_path/raw.
        split: Dataset split to load. One of "train", "test", "val", or "all".
            Splits are computed at runtime using the seed for reproducibility.
        transform: Optional transform. Only used if falling back to TrashData.
        fraction: Fraction of data to use (0.0-1.0). Must match preprocessed data.
        seed: Random seed for split generation. Must match preprocessing seed.

    Attributes:
        data_path: Root directory path for data storage.
        split: The dataset split being used.
        fallback_dataset: TrashData instance if preprocessed file not found.
        images: Preprocessed image tensors (shared across all splits).
        labels: Preprocessed labels (shared across all splits).
        split_indices: Indices for the requested split.
        classes: List of class names.
        class_to_idx: Mapping from class names to indices.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: Literal["train", "test", "val", "all"] = "train",
        transform: transforms.Compose | None = None,
        fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize the preprocessed TrashNet dataset.

        Loads preprocessed data from disk. If not available, falls back to TrashData.

        Args:
            data_path: Root directory for data.
            split: Dataset split to load ("train", "test", "val", or "all").
            transform: Optional custom transforms (only used for fallback).
            fraction: Fraction of data to use (0.0-1.0).
            seed: Random seed for reproducible splits.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.seed = seed
        self.fraction = fraction
        self.default_dataset = None

        # Check if preprocessed file exists
        self.preprocessed_file = self.data_path / "processed" / "trashnet.pt"

        if not self.preprocessed_file.exists():
            logger.warning(f"Preprocessed file not found at {self.preprocessed_file}")
            logger.warning("Falling back to TrashData (on-demand loading)")
            logger.info(f"To use preprocessed data, run: python -m trashsorting.data {self.data_path} --fraction {fraction}")

            # Fall back to TrashData
            raw_path = self.data_path / "raw"
            self.default_dataset = TrashData(
                raw_path, split=split, transform=transform,
                fraction=fraction, seed=seed
            )
            self.classes = self.default_dataset.classes
            self.class_to_idx = self.default_dataset.class_to_idx
            return

        # Try to load preprocessed data
        try:
            self._load_preprocessed()

            # Check if fraction matches
            if abs(self.metadata["fraction"] - fraction) > 0.001:
                logger.warning(f"Requested fraction {fraction} differs from preprocessed fraction {self.metadata['fraction']}")
                logger.warning(f"Falling back to TrashData with fraction {fraction}")
                logger.info(f"To use preprocessed data, run: python -m trashsorting.data {self.data_path} --fraction {fraction}")

                # Fall back to TrashData
                raw_path = self.data_path / "raw"
                self.default_dataset = TrashData(
                    raw_path, split=split, transform=transform,
                    fraction=fraction, seed=seed
                )
                self.classes = self.default_dataset.classes
                self.class_to_idx = self.default_dataset.class_to_idx
                return

            # Warn if transform is provided (not used for preprocessed data)
            if transform is not None:
                logger.warning("transform parameter is ignored for preprocessed data")
                logger.warning("Transforms are already applied during preprocessing")

            # Compute split indices
            total_size = len(self.images)
            self.split_indices, train_size, val_size, test_size = compute_splits(
                total_size, split, seed
            )

            if split == "all":
                logger.info(f"Loaded {total_size} samples from preprocessed data")
            else:
                split_size = len(self.split_indices) if self.split_indices is not None else 0
                logger.info(f"Loaded {split_size} {split} samples from preprocessed data")

        except Exception as e:
            logger.error(f"Error loading preprocessed file: {e}")
            logger.warning("Falling back to TrashData (on-demand loading)")

            # Fall back to TrashData
            raw_path = self.data_path / "raw"
            self.default_dataset = TrashData(
                raw_path, split=split, transform=transform,
                fraction=fraction, seed=seed
            )
            self.classes = self.default_dataset.classes
            self.class_to_idx = self.default_dataset.class_to_idx

    def _load_preprocessed(self) -> None:
        """Load preprocessed data from file."""
        data = torch.load(self.preprocessed_file, weights_only=False)

        # Validate structure
        if not all(key in data for key in ["images", "labels", "metadata"]):
            raise ValueError("Invalid preprocessed file format")

        self.images = data["images"]
        self.labels = data["labels"]
        self.metadata = data["metadata"]

        # Extract class information
        self.classes = self.metadata["classes"]
        self.class_to_idx = self.metadata["class_to_idx"]

    def __len__(self) -> int:
        """Return the number of samples in this split.

        Returns:
            Number of samples in the dataset split.
        """
        if self.default_dataset is not None:
            return len(self.default_dataset)

        if self.split == "all":
            return len(self.images)
        return len(self.split_indices) if self.split_indices is not None else 0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return a preprocessed sample from the dataset.

        Args:
            index: Index of the sample to retrieve (relative to split).

        Returns:
            Tuple of (image_tensor, label) where image_tensor is a
            torch.Tensor of shape (3, 224, 224) and label is an integer
            class index.
        """
        if self.default_dataset is not None:
            return self.default_dataset[index]

        # Map split-relative index to absolute index
        if self.split == "all":
            actual_index = index
        else:
            actual_index = self.split_indices[index] # type: ignore

        # Return preprocessed data
        return self.images[actual_index], int(self.labels[actual_index])


def compute_splits(
    total_size: int,
    split: Literal["train", "test", "val", "all"],
    seed: int = 42
) -> Tuple[Optional[List[int]], int, int, int]:
    """Compute train/val/test split indices.

    Generates reproducible 70/15/15 splits using a seeded random permutation.

    Args:
        total_size: Total number of samples in the dataset.
        split: Which split to return indices for.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (indices_for_split, train_size, val_size, test_size).
        indices_for_split is None if split="all", otherwise a list of indices.
    """
    if split == "all":
        return None, total_size, 0, 0

    # 70/15/15 split
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Generate split indices using torch.Generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    # Partition indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Return appropriate indices based on split
    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    return split_indices[split], train_size, val_size, test_size


def preprocess(
    data_path: Path = typer.Argument(..., help="Root directory for data (e.g., 'data/')"),
    fraction: float = typer.Option(1.0, help="Fraction of data to preprocess (0.0-1.0)"),
    seed: int = typer.Option(42, help="Random seed for shuffling"),
) -> None:
    """CLI command to preprocess the TrashNet dataset into a single file.

    The preprocessed data will be saved to data_path/processed/trashnet.pt

    Args:
        data_path: Root directory for data. Raw data should be in data_path/raw.
        fraction: Fraction of data to preprocess (0.0-1.0). Default 1.0.
        seed: Random seed for shuffling. Default 42.
    """
    # Set up logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    logger.info("Preprocessing TrashNet dataset...")

    raw_path = data_path / "raw"
    processed_path = data_path / "processed"

    # Create a temporary dataset just to call preprocess
    dataset = TrashData(raw_path, split="train", fraction=fraction)
    dataset.preprocess(processed_path, fraction=fraction, seed=seed)

    logger.info(f"✓ Preprocessing complete! Data saved to {processed_path / 'trashnet.pt'}")


if __name__ == "__main__":
    typer.run(preprocess)
