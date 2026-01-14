"""Prediction utilities for trash image classification."""

from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import torch
from PIL import Image
import typer
import logging

from trashsorting.model import TrashModel
from trashsorting.utils.transform import image_transform

logger = logging.getLogger(__name__)

# TrashNet class names
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def predict(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Classify the image into the recycling categories.

    Args:
        image_path: Path to the image file
        checkpoint_path: Path to the model checkpoint (.ckpt)
        class_names: Optional list of class names (defaults to TrashNet classes)

    Returns:
        Dictionary with prediction results containing:
            - predicted_class: The predicted class name
            - confidence: Confidence score (0-1)
            - all_probabilities: Dictionary of all class probabilities
            - predicted_idx: Index of the predicted class
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Load model (set pretrained=False to avoid downloading weights from HuggingFace)
    logger.info(f"Loading model from {checkpoint_path}")
    model = TrashModel.load_from_checkpoint(str(checkpoint_path), pretrained=False)
    model.eval()

    # Load and preprocess image
    logger.info(f"Loading image from {image_path}")
    image = Image.open(image_path)

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transforms
    transform = image_transform()
    transformed: torch.Tensor = transform(image)  # type: ignore[assignment]
    image_tensor = transformed.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = probabilities[predicted_idx].item()

    # Get all class probabilities
    all_probabilities = {
        class_names[i]: probabilities[i].item() for i in range(len(class_names))
    }

    result = {
        "predicted_class": class_names[predicted_idx],
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "predicted_idx": predicted_idx,
    }

    logger.info(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")

    return result


def predict_from_tensor(
    image_tensor: torch.Tensor,
    model: TrashModel,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Predict from a preprocessed image tensor.

    Useful when working with datasets that already provide tensors.

    Args:
        image_tensor: Image tensor (C, H, W) - single image only
        model: Loaded TrashModel instance
        class_names: Optional list of class names (defaults to TrashNet classes)

    Returns:
        Dictionary with prediction results

    Raises:
        ValueError: If image_tensor has batch dimension (4D tensor)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Check dimensions - only accept single images
    if image_tensor.dim() == 4:
        raise ValueError(
            f"Expected single image tensor with 3 dimensions (C, H, W), "
            f"got batch tensor with 4 dimensions {tuple(image_tensor.shape)}. "
            f"This function only processes single images."
        )

    if image_tensor.dim() != 3:
        raise ValueError(
            f"Expected image tensor with 3 dimensions (C, H, W), "
            f"got {image_tensor.dim()} dimensions {tuple(image_tensor.shape)}"
        )

    # Add batch dimension for model
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities).item())
    confidence = probabilities[predicted_idx].item()

    all_probabilities = {
        class_names[i]: probabilities[i].item() for i in range(len(class_names))
    }

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "predicted_idx": predicted_idx,
    }


def main(
    image_path: Path = typer.Argument(..., help="Path to image file"),
    checkpoint_path: Path = typer.Argument(..., help="Path to model checkpoint (.ckpt)"),
    verbose: bool = typer.Option(True, help="Show detailed probabilities"),
) -> None:
    """CLI command to predict the class of a trash image.

    Example:
        python -m trashsorting.predict path/to/image.jpg models/best.ckpt
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Make prediction
    result = predict(image_path, checkpoint_path)

    if verbose:
        for class_name, prob in sorted(
            result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            logger.debug(f"  {class_name:10s}: {prob:.1%}")


if __name__ == "__main__":
    typer.run(main)
