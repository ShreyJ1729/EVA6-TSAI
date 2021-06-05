# Session 5 - Coding Drill Down

## Goal
Write a neural network in three iterations, writing target, results, and analysis for each.

## Submission Info
### Iteration 1:
Target:
- Get basic working code

Results:
- 98.75% accuracy
- ~23k parameters (too many)
- Model converged quite slowly (first epoch only 11% training accuracy)
- Minor overfitting near last epochs

Analysis:
- Add BatchNorm for faster convergence
- Reduce model number of parameters

Training logs (last 5 epochs):

```
[Epoch 10]
Train set: Average loss: 0.0538, Accuracy: 59011/60000 (98.35%)
Test set: Average loss: 0.0583, Accuracy: 9813/10000 (98.13%)

[Epoch 11]
Train set: Average loss: 0.0490, Accuracy: 59090/60000 (98.48%)
Test set: Average loss: 0.0464, Accuracy: 9866/10000 (98.66%)

[Epoch 12]
Train set: Average loss: 0.0440, Accuracy: 59190/60000 (98.65%)
Test set: Average loss: 0.0397, Accuracy: 9875/10000 (98.75%)

[Epoch 13]
Train set: Average loss: 0.0419, Accuracy: 59205/60000 (98.67%)
Test set: Average loss: 0.0455, Accuracy: 9859/10000 (98.59%)

[Epoch 14]
Train set: Average loss: 0.0404, Accuracy: 59237/60000 (98.73%)
Test set: Average loss: 0.0471, Accuracy: 9846/10000 (98.46%)
```

### Iteration 2:
Target:
- \>99% Test Accuracy
- Under 10k parameters

Results:
- 99.16% accuracy (didn't stick till the end)
- ~7.5k parameters
- Accuracy jumping around near the end
- Overfitting in the last few epochs

Analysis:
- BN worked. First iteration now has 96% test acuracy.
- Need an LR-Scheduler to optimize lr
- Need dropout for regularizing overfitting

```
[Epoch 10]
Train set: Average loss: 0.0236, Accuracy: 59550/60000 (99.25%)
Test set: Average loss: 0.0324, Accuracy: 9892/10000 (98.92%)

[Epoch 11]
Train set: Average loss: 0.0230, Accuracy: 59573/60000 (99.29%)
Test set: Average loss: 0.0330, Accuracy: 9897/10000 (98.97%)

[Epoch 12]
Train set: Average loss: 0.0215, Accuracy: 59593/60000 (99.32%)
Test set: Average loss: 0.0284, Accuracy: 9916/10000 (99.16%)

[Epoch 13]
Train set: Average loss: 0.0187, Accuracy: 59655/60000 (99.42%)
Test set: Average loss: 0.0287, Accuracy: 9913/10000 (99.13%)

[Epoch 14]
Train set: Average loss: 0.0182, Accuracy: 59662/60000 (99.44%)
Test set: Average loss: 0.0327, Accuracy: 9892/10000 (98.92%)
```

### Iteration 3:
Target:
- 99.4% Test Accuracy (sticks till the end)
- Under 10k parameters

Results:
- 99.4% accuracy (stuck till the end)
- ~10k parameters

Analysis:
- LR scheduler worked beautifully near the end
- Even with dropout value=0.01, there was minor underfitting. Might be because of low number of model parameters.

```
[Epoch 10]
Train set: Average loss: 0.0247, Accuracy: 59540/60000 (99.23%)
Test set: Average loss: 0.0202, Accuracy: 9936/10000 (99.36%)

[Epoch 11]
Train set: Average loss: 0.0236, Accuracy: 59565/60000 (99.28%)
Test set: Average loss: 0.0182, Accuracy: 9940/10000 (99.40%)

[Epoch 12]
Train set: Average loss: 0.0224, Accuracy: 59577/60000 (99.30%)
Test set: Average loss: 0.0176, Accuracy: 9948/10000 (99.48%)

[Epoch 13]
Train set: Average loss: 0.0225, Accuracy: 59579/60000 (99.30%)
Test set: Average loss: 0.0185, Accuracy: 9940/10000 (99.40%)

[Epoch 14]
Train set: Average loss: 0.0218, Accuracy: 59604/60000 (99.34%)
Test set: Average loss: 0.0184, Accuracy: 9943/10000 (99.43%)
```

### Final Model Architecture


Colab notebook links are in the submission on canvas.