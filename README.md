# Efficient keyword spotting using Deep Learning
This repository contains the materials for Lab 1 of the Intelligent Architectures course (5LIL0). This aim of this assignment is to introduce students to the concepts of model design and optimization. 

## Installation
See requirements.txt for Python3 dependencies. Install dependencies using `python3 -m pip install --user -r requirements.txt`.

## Folder Structure
The project structure is adopted from `https://github.com/victoresque/pytorch-template`. The most important directories are explained below:
  ```
  ./
  │
  ├── train.py - main script to start training/finetuning
  ├── quantize.py - script to perform post-training quantization
  ├── prune.py - script to perform post-training pruning
  ├── test.py - evaluation of trained/quantized/pruned/finetuned model
  ├── profile.py - script to collect network statistics (e.g. #MACs, Model Size)
  │
  ├── config_lenet5_keywords.yml - holds configuration for training and optimization
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── data/ - directory where dataset is stored
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboardX and logging output
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo contains a baseline example on the Speech Recognition dataset. Try 

  ```
  python3 train.py -c config_lenet5_keywords.yml
  ```

to train a simple model. After training you can test the model with

  ```
  python3 test.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth
  ```
which should give you a similar accuracy as shown below:
```console
Loading checkpoint: saved/models/KeywordLeNet5Clone_exp1/0906_191755/model_best.pth ...
Predicted     0    1    2    3   All
True                                
0          2411   26   65   51  2553
1            80  337    0    0   417
2            92    0  252    4   348
3            65    0    0  289   354
All        2648  363  317  344  3672
Metrics: {'loss': 0.2813729564283928, 'top1': 0.8956971677559913}
```
The confusion matrix shows you how the model performed on different classes. As can be seen, most of the samples in this test set are in the zero-class (background noise).

To establish an optimization baseline you can run

  ```
  python3 profile.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth
  ```
which will print some relevant performance metrics
```console
...
MACs/classification: 170864
Number of Parameters: 12764
Model Size:
  - Weights: 398kB
  - Activations: 306kB
Accuracy: 89.56%
```

### Quantization
After training the model can be quantized. Weight and activation quantization helps to reduce the size of the model and intermediate results. Additionally it enables more efficient integer-based inference, compared to the standard single-precision floating-point model. Quantization might also result in speedups if, for example, the reference implementation was memory-bound. Additionally, reductions in bitwidth can potentially be exploited by specialized (vector-)processors, such as the ARM NEON coprocessor, which can perform 4x32b, 8x16b or 16x8b operations in parallel.

The quantization procedure aims to reduce the bitwidth of weights and activations until a user-defined error margin is exceeded. To quantize a model try
  ```
  python3 quantize.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth
  ```

which results in the following output:

```console
...
---------------------------------------------
Network quantization results.
Baseline float32: 0.90323
Conv layer weights:
    bitwidth: 16 => top1: 0.90348
    bitwidth:  8 => top1: 0.88827
    bitwidth:  4 => top1: 0.79958
    bitwidth:  2 => top1: 0.68302
FC layer weights:
    bitwidth: 16 => top1: 0.90323
    bitwidth:  8 => top1: 0.89917
    bitwidth:  4 => top1: 0.78620
    bitwidth:  2 => top1: 0.10613
Conv/FC inputs/activations:
    bitwidth: 16 => top1: 0.90348
    bitwidth:  8 => top1: 0.89693
    bitwidth:  4 => top1: 0.55120
Dynamic fixed point net:
Conv weights bitwidth: 4
FC weights bitwidth: 4
Activations bitwidth: 8
Accuracy: 0.86599
---------------------------------------------
Saving config: saved/models/KeywordLeNet5Clone_exp1/.../quantization.yml ...

```

A 4-bit fixed-point data format has been chosen for weights convolutional layers, and an 8-bit format is chosen for the activations of all layers and weights in fully-connected layers. The final top-1 validation accuracy of the reduced-precision network is 86.60%, which is a bit worse than the initial 90.32% of the full-precision model.

The quantization solution is saved in a `quantization.yml` file. To apply the test set to this solution try

  ```
  python3 test.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth \
    -q saved/models/KeywordLeNet5Clone_exp1/.../quantization.yml
  ```

The accuracy on the test set should be approximately the same as the validation set accuracy.


It should be noted that the inference is not actually performed in fixed-point integers. Instead, the range and precision (step size) of weights and inputs is reduced before and after the convolutional operator. The implementation of these simulated fixed-point layers can be found in `optimization/utility/layers.py`.

The quantization procedure is inspired by the approach of [P. Gysel et al. (2016)](https://arxiv.org/abs/1604.03168).


### Pruning
Pruning aims to the model size by removing non-important weights. In this assignment we use a post-training kernel-pruning method. Kernel-pruning means that complete kernels are being removed. Removing complete kernels results in a speedup that is almost directly proportional to the removed workload.

To prune the trained model we execute
    ```
    python3 prune.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth
    ```
which results in the following output

```console
...

Error margin exceeded; save previous solution!

[PRUNING] Step 3: Evaluate final solution on validation set.

final validation accuracy: 0.87220

---------------------------------------------
Network pruning results.
Baseline float32: 0.89844
Kernel pruning results:
Layer conv1    | Conv     | 0/8 kernels pruned   | 0.00% parameters pruned
Layer conv2    | Conv     | 1/8 kernels pruned   | 12.50% parameters pruned
Layer fc3      | Linear   | 32/64 kernels pruned | 50.00% parameters pruned
Layer fc4      | Linear   | 0/4 kernels pruned   | 0.00% parameters pruned
Total pruning rate: 47.00%
Accuracy: 0.87220
---------------------------------------------
```

The pruning method iteratively removes kernels until the user-defined error margin is exceeded. In this example 47% of all weights could be removed within a 3% error penalty.

The quantization solution is saved in a `pruning.yml` file. To apply the test set to this solution try

  ```
  python3 test.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth \
    -p saved/models/KeywordLeNet5Clone_exp1/.../pruning.yml
  ```
or
  ```
  python3 test.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth \
    -q saved/models/KeywordLeNet5Clone_exp1/.../quantization.yml \
    -p saved/models/KeywordLeNet5Clone_exp1/.../pruning.yml
  ```
to combine the pruning solution with the quantization solution.

The accuracy on the test set should be approximately the same as the validation accuracy.

The pruning method is based on the minimum weight method from [Molchanov et al. (2017)](https://arxiv.org/abs/1611.06440).

### Finetuning

The gains of performing quantization and pruning can be evaluated using the profiling script i.e.

  ```
  python3 profile.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth \
	-q saved/models/KeywordLeNet5Clone_exp1/.../quantization.yml \
	-p saved/models/KeywordLeNet5Clone_exp1/.../pruning.yml
  ```
leads to something like
```console
...
MACs/classification: 160114
Number of Parameters: 7908
Model Size:
  - Weights: 30kB
  - Activations: 75kB
Accuracy: 84.75%
  ```

As can be seen, the number of MACs and Model Size is reduced significantly. However, in the process accuracy suffered quite a bit.

We can deploy finetuning or retraining to recover some of the lost accuracy during quantization and pruning. Try

  ```
python3 train.py -r saved/models/KeywordLeNet5Clone_exp1/.../model_best.pth \
	-q saved/models/KeywordLeNet5Clone_exp1/.../quantization.yml \
	-p saved/models/KeywordLeNet5Clone_exp1/.../pruning.yml
  ```
  
Finetuning will retrain the remaining weights of the pruned network. Fixed-point finetuning updates the weights based on a quantized forward pass. The backwards pass remains high-precision to allow the small gradients to not get lost in the quantization noise.

### Deployment
The procedures in this repository simulate the behavior of a pruned network by masking zero-ed kernels or simulating fixed-point arithmetic. However, the resulting solutions can be converted to integer-based solutions for mobile and embedded platforms. Since the mapping procedure from PyTorch models to integer-based fixed-point code is mostly a cumbersome engineering task, we did not cover these things.
