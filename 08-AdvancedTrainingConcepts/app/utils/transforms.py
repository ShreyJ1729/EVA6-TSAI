import albumentations as A
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.transforms import Cutout
from albumentations.pytorch import ToTensorV2


def transforms(dataset_mean: int, dataset_std: int) -> dict:
    """
    A function to apply image transformation using Albumentation library.
    ---------------------------------------------------------------------

        - Input: Mean and standard deviation of the dataset.
        - Output: A dictonary with image transformations.
    """
    return {
        "train": A.Compose(
            [
                A.Rotate(limit=5),
                A.RandomCrop(height=32, width=32, p=0.5),
                A.PadIfNeeded(min_height=32 + 4, min_width=32 + 4, value=dataset_mean,),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=dataset_mean, std=dataset_std),
                ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [A.Normalize(mean=dataset_mean, std=dataset_std), ToTensorV2(),]
        ),
    }
