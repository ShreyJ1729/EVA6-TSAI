import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def loss_acc_curves(loss_stats: dict, acc_stats: dict):
    """
    Plot loss and accuracy curves.
    ------------------------------
        - Input: 2 dictionaries which contain the loss and accuracy
                 values per epoch. Each dictionary has the keys --
                 "train" and "val".
        - Output: Line plot.

    """
    train_val_acc_df = (
        pd.DataFrame.from_dict(acc_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epochs"})
    )
    train_val_loss_df = (
        pd.DataFrame.from_dict(loss_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epochs"})
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    sns.lineplot(
        data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]
    ).set_title("Train-Val Accuracy/Epoch")

    sns.lineplot(
        data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]
    ).set_title("Train-Val Loss/Epoch")


def score_report(y_true_list: list, y_pred_list: list, idx2class: dict):
    """
    Generate accuracy score, classification-report, and confusion matrix.
    ---------------------------------------------------------------------
        - Input: The true and predicted values along with a dictionary to
                 convert idx to class.
        - Output: Printed score report.
    """
    print(f"Test Accuracy = {accuracy_score(y_true_list, y_pred_list)}\n")
    print("=" * 50)
    print(
        f"\nClassification Report: \n\n{classification_report(y_true_list, y_pred_list)}\n\n"
    )
    print("=" * 50)
    print(f"\nConfusion Matrix: \n\n{confusion_matrix(y_true_list, y_pred_list)}\n")

    # plot confusion matrix
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_true_list, y_pred_list)
    ).rename(columns=idx2class, index=idx2class)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax).set(
        title="Confusion Matrix Heatmap"
    )

