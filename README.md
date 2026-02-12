# Awesome Data Selection

A curated list of papers and resources on data selection for machine learning.

---

## Contents

- [Active Learning](#active-learning)
- [Core-set Selection](#core-set-selection)
- [Curriculum Learning](#curriculum-learning)
- [Online Batch Selection](#online-batch-selection)
- [Rho-Loss Based](#rho-loss-based)
- [Importance Sampling](#importance-sampling)

---

## Introduction

Data selection studies how choosing *which data to use, when to use it, and how to weight it* affects learning efficiency, robustness, and generalization. Instead of treating all samples equally, modern machine learning increasingly recognizes data as an optimization variable.

### Curriculum Learning

Curriculum Learning proposes training models by organizing examples from easier to harder, improving optimization stability and convergence. Rather than removing data, it schedules exposure to examples in a structured progression, inspired by human education.

### Importance Sampling vs Curriculum Learning

Importance Sampling reweights or resamples examples based on gradient contribution or variance reduction. Unlike Curriculum Learning, it does not impose a difficulty order, but instead focuses on statistical efficiency during stochastic optimization.

### Active Learning

Active Learning selects the most informative unlabeled samples for annotation to reduce labeling costs. It typically relies on uncertainty, disagreement, expected model change, or diversity-based criteria to decide which data points to query next.

### Core-set Selection

Core-set methods aim to extract a representative subset of a labeled dataset while preserving performance. They often rely on geometric coverage, embedding diversity, clustering, or submodular optimization to approximate full-dataset training.

### Proxy-Based Selection

Proxy methods use smaller or cheaper surrogate models to approximate sample importance. These proxies estimate influence, difficulty, or gradient impact before expensive large-scale training.

### Rho-Loss Based Selection

Rho-loss methods rely on robust loss functions to identify noisy, harmful, or outlier samples. By analyzing sample-wise loss curvature, they downweight or remove detrimental data points.

### Bayesian Treatment for Rho-Loss

Bayesian approaches model uncertainty in loss contributions, providing probabilistic interpretations of sample reliability. This enables principled filtering under noise and improves robustness in large-scale training.

---

# Methods (20-word descriptions)

### Active Learning
Selects the most informative unlabeled samples for annotation, minimizing labeling cost while maximizing expected model performance improvement.

### Core-set Selection
Identifies representative subsets of labeled datasets using geometric coverage or diversity to maintain performance with fewer samples.

### Curriculum Learning
Orders training samples by difficulty, presenting easier examples first to stabilize optimization and accelerate convergence.

### Online Batch Selection
Dynamically reweights or resamples training examples during optimization based on loss, gradients, or uncertainty.

### Rho-Loss Based
Uses robust loss functions to detect noisy, harmful, or low-quality samples for downweighting or removal.

### Importance Sampling
Samples training data proportionally to gradient magnitude or variance contribution to improve stochastic optimization efficiency.

---

# Papers

## 2024 – 2022

### Data Pruning via Gradient Signal Preservation (2023)
Proposes selecting samples that preserve the global gradient direction of the full dataset. Demonstrates significant dataset reduction without major performance loss in deep networks.

### Efficient Data Selection for Large Language Models (2023)
Introduces scalable filtering strategies for LLM pretraining using perplexity and loss-based scoring. Shows compute-efficient pretraining with minimal degradation.

### Dataset Distillation by Matching Training Trajectories (2022)
Optimizes synthetic subsets by aligning full training trajectories. Demonstrates strong compression of datasets into small synthetic coresets.

---

## 2020 – 2022

### Forgetting Events in Deep Learning (Toneva et al., 2019)
Analyzes how often samples are forgotten during training. Shows that frequently forgotten samples are more informative and can guide pruning strategies.

### Data Shapley: Equitable Valuation of Data (2019)
Applies Shapley value concepts to quantify each sample’s contribution to model performance. Establishes principled foundations for data valuation and removal.

### GradMatch: Gradient Matching based Data Subset Selection (2021)
Selects subsets by minimizing gradient difference between full and reduced datasets. Provides scalable subset selection with theoretical grounding.

---

## 2020 – 2018

### Core-Set Approach for Active Learning (Sener & Savarese, 2018)
Formulates active learning as a core-set problem in feature space. Uses k-center approximation to select diverse and representative samples.

### Curriculum Learning (Bengio et al., 2009 – revived in deep era)
Introduces structured training progression from easy to hard samples. Shows faster convergence and improved generalization.

### Online Batch Selection for Faster Training (2018)
Prioritizes high-loss samples dynamically during training. Demonstrates improved convergence speed in deep networks.

---

## 2018 and Earlier

### Importance Sampling for Stochastic Gradient Descent (Needell et al., 2014)
Analyzes variance reduction through non-uniform sampling. Provides theoretical guarantees for faster convergence in SGD.

### Self-Paced Learning (Kumar et al., 2010)
Learns curriculum automatically by progressively incorporating harder samples. Bridges optimization and curriculum strategies.

### Robust Loss Functions for Noisy Labels (Various Early Works)
Introduces alternative loss functions resilient to mislabeled data. Forms the foundation for modern rho-loss and robust training methods.

---
