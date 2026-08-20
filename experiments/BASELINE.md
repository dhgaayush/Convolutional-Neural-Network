# Baseline Experiment

## Experiment ID

baseline_v1

## Objective

Establish a reproducible performance baseline for the C++ Convolutional Neural Network implementation before hyperparameter or architecture experimentation.

---

## Dataset

Dataset: MNIST

Source: Official MNIST dataset

Preparation pipeline:

scripts/prepare_mnist.py

Training samples: 5,000

Test samples: 1,000

Image dimensions: 28 × 28

Classes: 10

---

## Model

Architecture:

Input Image
    ↓
Convolutional Layers
    ↓
Pooling Layers
    ↓
Flatten
    ↓
Dense Layer: 256 neurons
    ↓
Output Layer: 10 neurons

Neural network topology:

{ inputSize, 256, 10 }

---

## Hyperparameters

Neural Network Learning Rate:

0.001

Convolution Learning Rate:

0.00001

Epochs:

15

Convolution Learning Rate Decay:

Every 5 epochs:

currentConvLR *= 0.5

Initial convolution learning rate:

0.00001

After epoch 5:

0.000005

After epoch 10:

0.0000025

---

## Results

Training Accuracy:

93.4%

Training Loss:

0.224133

Training Correct:

4670 / 5000

Test Accuracy:

88.7%

Test Correct:

887 / 1000

Generalization Gap:

4.7 percentage points

---

## Artifacts

Trained model:

my_model.bin

Final checkpoint:

checkpoint_epoch_15.bin

---

## Notes

This experiment serves as the reference baseline.

Future experiments should modify one primary variable at a time where possible and compare results against this baseline.

