# Enhancing Fine-grained Urban Flow Inference via Progressive
Multimodal Learning

This repository provides a reference implementation of the paper: *Enhancing Fine-grained Urban Flow Inference via Progressive
Multimodal Learning*
## Overview

​                                                                        <img alt="overview" height="400" src="image/framework.png" width="400"/>

Urban flow inference, especially when based on coarse-grained or partially observed data, plays a crucial role in developing resilient and sustainable urban mobility systems. While advancing
spatial modeling and external factor integration, existing methods face limitations in capturing cross-modal structural dependencies, ensuring robustness under partial or noisy observations, and adapting training paradigms to evolving or degraded data conditions.
To address these limitations, we propose Progressive Multimodal Learning (PML), a unified framework that performs masked multimodal representation learning with progressive optimization for robust fine-grained urban flow inference. The PML integrates spatial flow maps, road topology, temporal indicators, and environmental variables through a hierarchical encoder that captures both global context and localized spatial dynamics. During pretraining, a spatial masking strategy is employed to simulate partial observability and guide the model toward learning structured spatial representations.
The training process follows a progressive multi-phase scheme that includes masked self-supervised pretraining, modality-aligned
supervised learning, and task-specific fine-tuning. Extensive experiments on large-scale urban datasets demonstrate that the proposed
approach consistently surpasses state-of-the-art baselines in predictive accuracy and robustness under degraded input conditions,
confirming the effectiveness of progressive masked multimodal learning for real-world urban forecasting.

## Requirements
We implement ENHANCER and other FUFI methods with the following dependencies:
* python 3.11.5
* pytorch 2.1.2
* einops
* scikit-learn

## Datasets
TaxiBJ datasets can be obtained from the baseline [UrbanFM's repository](https://github.com/yoshall/UrbanFM/tree/master/data).

## Usage
Before running the code, ensure the package structure of ENHANCER is as follows:
```
.
├── datasets
│   └── TaxiBJ
│       ├── P1
│       ├── P2
│       ├── P3
│       └── P4
│   └── Chengdu
│   └── Xian
├── experiments
├── model
├── model_train
└── utils_pack
```



## Citing

```

```