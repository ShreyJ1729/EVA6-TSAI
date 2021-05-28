# Session 3 - PyTorch

## Goal
Write a neural network that has
- ### 2 inputs:
  - an image from MNIST dataset, and
  - a random number between 0 and 9
- ### 2 outputs:
  - the "number" that was represented by the MNIST image, and
  - the "sum" of this number with the random number that was generated and sent as the input to the network


## Approach Outline

- ### Created EnhancedMNIST Dataset to add the label for the random number and the sum
- ### NN Architecture
  - Conv layers _ log_softmax for MNIST classification
  - One-Hot-Vector output (OHV) of MNIST is concatenated with OHV of random number input
  - Concatenated vector is put through two fully-connected layers + log_softmax to get sum prediction (OHV)
  - Outputs MNIST OHV and sum prediction OHV
- ### Training
  - Sum prediction won't get better until MNIST prediction gets better, so until MNIST gets 50% accuracy, only train MNIST
  - Once MNIST hits 50%, alternate between training MNIST conv layers and sum fully-connected layers
  - Once MNIST > 90%, train only sum fully-connected layers
  - Due to random variations in the accuracy for each batch, MNIST may still train some batches even if MNIST epoch accuracy > 90%, as show in the training logs.
- ### Results
  - \> 99% test accuracy for both tasks within 5 epochs 

Colab Notebook Link: https://colab.research.google.com/github/ShreyJ1729/EVA6-TSAI/blob/main/03-PyTorch/3-PyTorch.ipynb