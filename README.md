# CRS: Conversational Recommender Systems

This repository contains the implementation and evaluation code for the paper "Model Meets Knowledge: Analyzing Knowledge Types for Conversational Recommender Systems". We provide a comprehensive collection of methods evaluated in our research, along with a unified dataset and standardized evaluation protocol.

## Overview

This repository contains the complete implementation for all four research questions addressed in our paper. Each research question has its own dedicated directory with specific baselines and evaluation protocols.

## Research Questions

### RQ1: Overall Evaluation
To reproduce the results for RQ1, navigate to each baseline directory in the root folder and execute:
```bash
sbatch train.sh
```

### RQ2: Model-Knowledge Compatibility
To reproduce the results for RQ2, navigate to each baseline directory (configured with different knowledge types) under the RQ2 directory and execute:
```bash
sbatch train.sh
```

### RQ3: Knowledge Complementarity
To reproduce the results for RQ3, navigate to each baseline directory (configured with different knowledge types) under the RQ3 directory and execute:
```bash
sbatch train.sh
```

### RQ4: Scenario-specific Analysis
Here, we provide the data process code to split data according to their dialogue scernarios. 
