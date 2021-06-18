# Session 7 - Advanced Concepts

## Goal
Write a neural network under 200k parameters that gets >85% test accuracy on CIFAR-10 using the specified transformations through the albumentations library.

## Submission Info

### Code Walkthrough
I used the albumentations library to perform the specified transforms:
```
image_transforms = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.25),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, fill_value=0.4733630004850904, mask_fill_value=None, p=0.5),
            A.Normalize(mean=(0.49139967861519745, 0.4821584083946076, 0.44653091444546616), std=(0.2470322324632823, 0.24348512800005553, 0.2615878417279641)),
            ToTensorV2(),
        ]
    )
```

Network Architecture:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
       BatchNorm2d-6           [-1, 64, 32, 32]             128
              ReLU-7           [-1, 64, 32, 32]               0
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 16, 16]           2,048
           Conv2d-10           [-1, 32, 16, 16]             288
      BatchNorm2d-11           [-1, 32, 16, 16]              64
             ReLU-12           [-1, 32, 16, 16]               0
          Dropout-13           [-1, 32, 16, 16]               0
           Conv2d-14             [-1, 64, 8, 8]           2,048
      BatchNorm2d-15             [-1, 64, 8, 8]             128
             ReLU-16             [-1, 64, 8, 8]               0
          Dropout-17             [-1, 64, 8, 8]               0
           Conv2d-18          [-1, 128, 12, 12]          73,728
      BatchNorm2d-19          [-1, 128, 12, 12]             256
             ReLU-20          [-1, 128, 12, 12]               0
          Dropout-21          [-1, 128, 12, 12]               0
           Conv2d-22             [-1, 64, 5, 5]          73,728
      BatchNorm2d-23             [-1, 64, 5, 5]             128
             ReLU-24             [-1, 64, 5, 5]               0
          Dropout-25             [-1, 64, 5, 5]               0
        AvgPool2d-26             [-1, 64, 1, 1]               0
           Conv2d-27            [-1, 256, 1, 1]          16,384
             ReLU-28            [-1, 256, 1, 1]               0
      BatchNorm2d-29            [-1, 256, 1, 1]             512
          Dropout-30            [-1, 256, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]           2,560
================================================================
Total params: 191,360
Trainable params: 191,360
Non-trainable params: 0
----------------------------------------------------------------
```

Since I'd be training the network for the long time I chose to use `torch.optim.lr_scheduler.ReduceLROnPlateau()`:
```
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
```

The network was trained for 399 epochs until 85% test accuracy was reached. Last 5 epochs are below.

```
[Epoch 395]
Train set: Accuracy: 44913/50000 (89.83%)
Test set: Accuracy: 8494/10000 (84.94%)

[Epoch 396]
Train set: Accuracy: 44881/50000 (89.76%)
Test set: Accuracy: 8402/10000 (84.02%)

[Epoch 397]
Train set: Accuracy: 44885/50000 (89.77%)
Test set: Accuracy: 8488/10000 (84.88%)

[Epoch 398]
Train set: Accuracy: 44887/50000 (89.77%)
Test set: Accuracy: 8494/10000 (84.94%)

[Epoch 399]
Train set: Accuracy: 44879/50000 (89.76%)
Test set: Accuracy: 8501/10000 (85.01%)
```

Colab link: https://colab.research.google.com/github/ShreyJ1729/EVA6-TSAI/blob/main/07-AdvancedConcepts/07-AdvancedConcepts.ipynb