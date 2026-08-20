# Data Volume Experiment — 10k

## Experiment ID

data_volume_10k

## Objective

Measure the effect of doubling the MNIST training dataset while keeping the model architecture and core hyperparameters unchanged.

---

## Dataset

Training samples: 10,000

Test samples: 1,000

Dataset: Official MNIST

Preprocessing:

scripts/prepare_mnist.py

---

## Controlled Variables

Epochs: 15

NN learning rate: 0.001

Initial convolution learning rate: 0.00001

Convolution LR decay: multiply by 0.5 every 5 epochs

Dense topology:

{ inputSize, 256, 10 }

---

## Result

Final Test Accuracy:

89.6%

Correct predictions:

896 / 1000

---

## Comparison with Baseline

Baseline training samples: 5,000

Baseline test accuracy: 88.7%

10k training samples test accuracy: 89.6%

Improvement:

+0.9 percentage points

Additional correct predictions:

+9 / 1000

---

## Interpretation

Doubling the training dataset from 5,000 to 10,000 samples improved test accuracy by 0.9 percentage points.

The model is still benefiting from additional data, though the improvement is smaller than the proportional increase in dataset size.

---

## Artifact

Trained model:

model_10k.bin
