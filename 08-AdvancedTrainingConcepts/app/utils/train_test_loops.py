import torch
from tqdm.notebook import tqdm


def multi_acc(y_pred: "torch.tensor", y_test: "torch.tensor") -> float:
    """
    Calculate Accuracy.
    -------------------
        - Input: Predicted and actual output values.
        - Output: Accuracy percentage.
    """
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def print_log(
    e,
    epochs,
    avg_train_epoch_loss,
    avg_val_epoch_loss,
    avg_train_epoch_acc,
    avg_val_epoch_acc,
):
    """
    Print training logs.
    """
    print(
        f"Epoch {e+0:02}/{epochs}: | Train Loss: {avg_train_epoch_loss:.5f} | Val Loss: {avg_val_epoch_loss:.5f} | Train Acc: {avg_train_epoch_acc:.3f}% | Val Acc: {avg_val_epoch_acc:.3f}%"
    )


def validate(model, criterion, val_loader, device):
    """
    Validation loop.
    """
    val_epoch_loss = 0
    val_epoch_acc = 0

    with torch.no_grad():
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = (
                X_val_batch.to(device),
                y_val_batch.to(device),
            )

            y_val_pred = model(X_val_batch).squeeze()

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

    return val_epoch_loss, val_epoch_acc


def train_loop(
    model, epochs, optimizer, criterion, scheduler, train_loader, val_loader, device
):
    """
    The main training loop.
    -----------------------

    This function trains the data on the train-dataset. Along with this, it tests
    the model on a validation dataset created out of the train-dataset.
    """

    if scheduler.__class__.__name__ not in [
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "ReduceLROnPlateau",
        "CyclicLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
    ]:
        raise ValueError("Unsupported learning rate scheduler.")

    acc_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    print("\nBegin training.")
    ##############################################################################
    # TRAIN
    ##############################################################################

    for e in tqdm(range(1, epochs + 1)):

        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
            X_train_batch, y_train_batch = (
                X_train_batch.to(device),
                y_train_batch.to(device),
            )

            optimizer.zero_grad()

            y_train_pred = model(X_train_batch).squeeze()

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            if scheduler.__class__.__name__ in ["CyclicLR", "OneCycleLR"]:
                scheduler.step()

            if scheduler.__class__.__name__ in ["CosineAnnealingWarmRestarts"]:
                scheduler.step(e + i / len(train_loader))

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        ##############################################################################
        # VALIDATE
        ##############################################################################
        val_epoch_loss, val_epoch_acc = validate(model, criterion, val_loader, device)

        if scheduler.__class__.__name__ in [
            "LambdaLR",
            "MultiplicativeLR",
            "StepLR",
            "MultiStepLR",
            "ExponentialLR",
        ]:
            scheduler.step()

        if scheduler.__class__.__name__ in ["ReduceLROnPlateau"]:
            scheduler.step(val_epoch_loss)

        ##############################################################################
        # LOGS
        ##############################################################################
        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        avg_train_epoch_acc = train_epoch_acc / len(train_loader)
        avg_val_epoch_acc = val_epoch_acc / len(val_loader)

        loss_stats["train"].append(avg_train_epoch_loss)
        loss_stats["val"].append(avg_val_epoch_loss)
        acc_stats["train"].append(avg_train_epoch_acc)
        acc_stats["val"].append(avg_val_epoch_acc)

        print_log(
            e,
            epochs,
            avg_train_epoch_loss,
            avg_val_epoch_loss,
            avg_train_epoch_acc,
            avg_val_epoch_acc,
        )

    return model, loss_stats, acc_stats


def test_loop(test_loader, model, device):
    """
    The test loop. This returns a tuple of lists that contain
    the true and predicted outputs.
    """
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_test_pred = model(x_batch)
            y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim=1)

            for i in y_pred_tag.squeeze().cpu().numpy().tolist():
                y_pred_list.append(i)

            for i in y_batch.squeeze().cpu().numpy().tolist():
                y_true_list.append(i)

    return y_pred_list, y_true_list
