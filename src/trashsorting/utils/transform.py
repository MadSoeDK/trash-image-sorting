from torchvision import transforms

def image_transform() -> transforms.Compose:
    return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize(),
                ]
            )


def normalize() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

def denormalize() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
