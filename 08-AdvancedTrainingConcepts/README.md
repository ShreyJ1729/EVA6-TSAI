# Assignment 8

This assignment is on GradCAM. The final notebook is `final_eva6_assignment_8.ipynb`.

The folder structure is as follows -

```
- app
    - models
    - datasets
    - explainibility
    - utils
- main.py
- final_eva6_assignment_8
```

The `models` folder contains the code for ResNet. The `datsets` folder contains the PyTorch dataset for CIFAR10. The `explainibility` folder contains code for GradCAM and `utils` folder contains all the other misc functions such as train/test loops, visualization, transformations etc.

## Image Transformations

![transformations](./assets/transformations..png)

## Loss Acc Curves

![lossacc_curves](./assets/lossacc_curves.png)

## Confusion Matrix

![cmatirx_heatmap](./assets/cmatirx_heatmap.png)

## Misclassified Examples

![misclassified](./assets/misclassified.png)

## GradCAM

![gradcam](./assets/gradcam.png)
