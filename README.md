# CRS: Conversational Recommender Systems

This repository contains the implementation and evaluation code for the paper "Model Meets Knowledge: Analyzing Knowledge Types for Conversational Recommender Systems". We provide a comprehensive collection of methods evaluated in our research, along with a unified dataset and standardized evaluation protocol.

## Overview

This repository contains the complete implementation for all four research questions addressed in our paper. Each research question has its own dedicated directory with specific baselines and evaluation protocols.

## Hardware Requirements

Our experiments were conducted on the following hardware configuration:
- **GPU**: NVIDIA A100 GPUs
- **CPU**: Intel Xeon CPUs (18 cores)
- **Memory**: 120 GiB


## Environment
```bash
conda env create -f environment.yml
```

## Research Questions

### RQ1: Overall Evaluation
To reproduce the results for RQ1, navigate to each baseline directory under the RQ1 directory and read the README.md file under that directory to know how to execute the code.

### RQ2: Model-Knowledge Compatibility
To reproduce the results for RQ2, navigate to each baseline directory (configured with different knowledge types) under the RQ2 directory and read the README.md file under that directory to know how to execute the code.

### RQ3: Knowledge Complementarity
To reproduce the results for RQ3, navigate to each baseline directory (configured with different knowledge types) under the RQ3 directory and read the README.md file under that directory to know how to execute the code.

### RQ4: Scenario-specific Analysis
Here, we provide the data process code to split data according to their dialogue scernarios. 
