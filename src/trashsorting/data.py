from pathlib import Path
from typing import Literal, Tuple

import torch
import typer
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from typing import Any, cast

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
        print(f"Loading TrashNet dataset (fraction: {fraction})...")
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

        # Split into train/val/test (70/15/15)
        if split == "all":
            self.dataset = full_dataset
        else:
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size

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
            # Using ImageNet normalization for easier transfer learning
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        print(f"Dataset loaded: {len(self)} samples across {len(self.classes)} classes")

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

    def preprocess(self, output_folder: str | Path) -> None:
        """Preprocess the raw data and save it to the output folder.

        This method iterates through the dataset, applies transforms, and
        saves the processed images and labels to disk for faster loading
        in subsequent training runs.

        Args:
            output_folder: Directory where preprocessed data will be saved.
                Typically 'data/processed' following Cookiecutter structure.
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Preprocessing {len(self)} samples to {output_path}...")

        # Save processed tensors
        images = []
        labels = []

        for idx in range(len(self)):
            image_tensor, label = self[idx]
            images.append(image_tensor)
            labels.append(label)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(self)} samples...")

        # Stack and save as PyTorch tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        split_name = self.split
        torch.save(images_tensor, output_path / f"{split_name}_images.pt")
        torch.save(labels_tensor, output_path / f"{split_name}_labels.pt")

        # Save metadata
        metadata = {
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "num_samples": len(self),
            "split": self.split,
        }
        torch.save(metadata, output_path / f"{split_name}_metadata.pt")

        print(f"Preprocessing complete! Saved to {output_path}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    """CLI command to preprocess the TrashNet dataset.

    Args:
        data_path: Root directory for raw dataset cache.
        output_folder: Directory where preprocessed data will be saved.
    """
    print("Preprocessing data...")
    dataset = TrashData(data_path, 'all', fraction=0.25)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
