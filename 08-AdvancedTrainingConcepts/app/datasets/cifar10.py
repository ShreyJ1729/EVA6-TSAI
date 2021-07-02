import numpy as np
from torchvision import datasets


def load_cifar10(
    dataset_path: str, is_train: bool, image_transforms: "Albumentation Transforms"
) -> "PyTorch Dataset":

    """
    Load CIFAR10 dataset using torchvision.
    ---------------------------------------

        - Input: dataset_path, is_train, and image_transforms.
        - Output: PyTorch Dataset object.
    """

    return datasets.CIFAR10(
        root=dataset_path,
        train=is_train,
        download=True,
        transform=lambda x: image_transforms(image=np.array(x))["image"],
    )
