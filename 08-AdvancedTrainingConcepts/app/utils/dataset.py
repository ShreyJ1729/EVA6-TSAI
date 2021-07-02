import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler


def calc_data_stats(dataset: "PyTorch Dataset") -> tuple:
    """
    Calculate dataset mean and standard deviation.
    -----------------------------------------------

        - Input: PyTorch Dataset Object.
        - Output: Tuple of mean and std.

    """
    np_train_dataset = dataset.data / 255

    mean_1, mean_2, mean_3 = (
        np_train_dataset[:, :, :, 0].mean(),
        np_train_dataset[:, :, :, 1].mean(),
        np_train_dataset[:, :, :, 2].mean(),
    )

    std_1, std_2, std_3 = (
        np_train_dataset[:, :, :, 0].std(),
        np_train_dataset[:, :, :, 1].std(),
        np_train_dataset[:, :, :, 2].std(),
    )

    return (mean_1, mean_2, mean_3), (std_1, std_2, std_3)


def visualise_transforms(
    original_data: "PyTorch Dataset",
    transformed_data: "PyTorch Dataset",
    num_samples: int,
):
    """
    Visualize the effects of image transformation.
    ----------------------------------------------

        - Input: Original dataset, transformed dataset, and number of samples to display.
        - Output: A graph using matplotlib.
    """

    images_original = [original_data[i][0] for i in range(num_samples)]
    images_transformed = [transformed_data[i][0] for i in range(num_samples)]

    plt.suptitle("Original vs Transformed Images")

    fig, axes = plt.subplots(figsize=(30, 10), nrows=2, ncols=num_samples)

    for i in range(num_samples):
        axes[0, i].imshow(images_original[i].permute(1, 2, 0))
        axes[0, i].title.set_text("OG")

    for i in range(num_samples):
        axes[1, i].imshow(images_transformed[i].permute(1, 2, 0))
        axes[1, i].title.set_text("TF")

    for ax in fig.axes:
        ax.axis("off")
        ax.grid("False")


def create_samplers(dataset: "PyTorch Dataset", train_percent: float):
    """
    Create train-val sampler for dataloader using SubsetRandomSampler.
    ------------------------------------------------------------------
        - Input: Pytorch Dataset and a number to denote the percentage
                 of samples for training data.
        - Output: Sampler.
    """
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))

    np.random.shuffle(dataset_indices)

    train_split_index = int(np.floor(train_percent * dataset_size))

    train_idx = dataset_indices[:train_split_index]
    val_idx = dataset_indices[train_split_index:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler


def class_to_idx(classes: list) -> dict:
    """
    Generate idx for each class starting from 0.
    -------------------------------------------
        - Input: List of classes.
        - Output: Dictionary of class-idx pairs.

    """
    return {c: i for i, c in enumerate(classes)}


def idx_to_class(class2idx: dict) -> dict:
    """
    Reverse the class-idx pair of a class2idx dictionary.
    -----------------------------------------------------
        - Input: class2idx dictionary.
        - Output: idx2class dictionary.
    """
    return {v: k for k, v in class2idx.items()}

