import math
import matplotlib.pyplot as plt
from app.explainability.gradcam import GradCAM


def get_misclassified_info(y_pred_list: list, y_true_list: list) -> list:
    """
    Get idx, predicted output, and true output of a model.
    ------------------------------------------------------
        - Input: Predicted and true output.
        - Output: A list of dicts of all misclassified samples.
    """
    return [
        {"idx": i, "pred": pred, "true": actual}
        for i, (pred, actual) in enumerate(zip(y_pred_list, y_true_list))
        if pred != actual
    ]


def visualize_misclassified_images(
    misclassified_info, dataset, idx_to_class, num_samples, row_img_limit=5
):
    if num_samples > row_img_limit:
        fig, axes = plt.subplots(
            figsize=(50, 20),
            nrows=math.ceil(num_samples / row_img_limit),
            ncols=row_img_limit,
        )
        for i in range(num_samples):
            axes[i // 5, i % 5].imshow(
                dataset[misclassified_info[i]["idx"]][0].permute(1, 2, 0)
            )
            axes[i // 5, i % 5].set_title(
                f"True: {idx_to_class[misclassified_info[i]['true']]}\nPred: {idx_to_class[misclassified_info[i]['pred']]}",
                fontsize=24,
            )
    else:
        fig, axes = plt.subplots(figsize=(30, 10), nrows=1, ncols=row_img_limit)
        for i in range(num_samples):
            axes[i].imshow(dataset[misclassified_info[i]["idx"]][0].permute(1, 2, 0))
            axes[i].title.set_text(
                f"True: {idx_to_class[misclassified_info[i]['true']]}\nPred: {idx_to_class[misclassified_info[i]['pred']]}",
                fontsize=24,
            )

    for ax in fig.axes:
        ax.axis("off")
        ax.grid("False")

    plt.suptitle("Misclassified Images", fontsize=32)


def misclassified_gradcam(
    model, misclassified_info, dataset, idx_to_class, num_samples, device
):

    fig, axes = plt.subplots(figsize=(5, 40), nrows=num_samples, ncols=2)
    for i in range(num_samples):
        axes[i, 0].imshow(dataset[misclassified_info[i]["idx"]][0].permute(1, 2, 0))
        axes[i, 0].set_title(
            f"True: {idx_to_class[misclassified_info[i]['true']]}\nPred: {idx_to_class[misclassified_info[i]['pred']]}",
            fontsize=10,
        )

        cam = GradCAM(
            model=model, target_layer=model.layer_2.base_layer[-1].base_block[-1][0],
        )
        output = cam(dataset[misclassified_info[i]["idx"]][0].to(device))
        axes[i, 1].imshow(output)

    for ax in fig.axes:
        ax.axis("off")
        ax.grid("False")

    plt.suptitle("GradCAM Misclassified Images", fontsize=18)
