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

### Most Influential Subset Selection: Challenges, Promises, and Beyond (2024)
Comprehensive analysis of approaches to identify the most influential subsets of training data. Examines failure modes of influence-based greedy heuristics and demonstrates the effectiveness of adaptive algorithms that capture sample interactions.

[📄 PDF Anotado](annotated_papers/(2024)%20most%20influential%20subset%20selection:%20challenges,%20promises,%20and%20beyond.pdf)

### Towards Accelerated Model Training via Bayesian Data Selection (2023)
Proposes Bayesian approaches to model uncertainty in loss contributions, providing probabilistic interpretations of sample reliability for principled filtering under noise.

[📄 PDF Anotado](annotated_papers/(2023)%20towards%20accelerated%20model%20training%20via%20bayesian.pdf)

### (RhoLoss) Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (2022)
Uses robust loss functions (rho-loss) to identify noisy, harmful, or outlier samples. Analyzes sample-wise loss curvature to downweight or remove detrimental data points.

[📄 PDF Anotado](annotated_papers/(2022)%20(rholoss)%20prioritized%20training%20on%20points%20that.pdf)

---

## 2020 – 2022

### Coresets via Bilevel Optimization for Continual Learning and Streaming (2020)
Identifies representative subsets of labeled datasets using geometric coverage or diversity to maintain performance with fewer samples, optimized via bilevel optimization for continual learning scenarios.

[📄 PDF Anotado](annotated_papers/(2020)%20coresets%20via%20bilevel%20optimization%20for%20continual%20learning%20and%20streaming.pdf)

### Selection via Proxy: Efficient Data Selection for Deep Learning (2020)
Uses smaller or cheaper surrogate models to approximate sample importance. These proxies estimate influence, difficulty, or gradient impact before expensive large-scale training.

[📄 PDF Anotado](annotated_papers/(2020)%20selection%20via%20proxy.pdf)

### Accelerating Deep Learning by Focusing on the Biggest Losers (2019)
Dynamically reweights or resamples training examples during optimization based on loss, gradients, or uncertainty, prioritizing high-loss samples to improve convergence speed.

[📄 PDF Anotado](annotated_papers/(2019)%20accelerating%20deep%20learning%20by%20focusing%20on%20the%20b.pdf)

### Not All Samples Are Created Equal: Deep Learning with Importance Sampling (2019)
Samples training data proportionally to gradient magnitude or variance contribution to improve stochastic optimization efficiency, demonstrating that not all samples contribute equally to learning.

[📄 PDF Anotado](annotated_papers/(2019)%20not%20all%20samples%20are%20created%20equal.pdf)

---

## 2020 – 2018

### Coresets via Bilevel Optimization for Continual Learning and Streaming (2020)
Identifies representative subsets of labeled datasets using geometric coverage or diversity to maintain performance with fewer samples, optimized via bilevel optimization for continual learning scenarios.

[📄 PDF Anotado](annotated_papers/(2020)%20coresets%20via%20bilevel%20optimization%20for%20continual%20learning%20and%20streaming.pdf)

### Selection via Proxy: Efficient Data Selection for Deep Learning (2020)
Uses smaller or cheaper surrogate models to approximate sample importance. These proxies estimate influence, difficulty, or gradient impact before expensive large-scale training.

[📄 PDF Anotado](annotated_papers/(2020)%20selection%20via%20proxy.pdf)

### Accelerating Deep Learning by Focusing on the Biggest Losers (2019)
Dynamically reweights or resamples training examples during optimization based on loss, gradients, or uncertainty, prioritizing high-loss samples to improve convergence speed.

[📄 PDF Anotado](annotated_papers/(2019)%20accelerating%20deep%20learning%20by%20focusing%20on%20the%20b.pdf)

### Not All Samples Are Created Equal: Deep Learning with Importance Sampling (2019)
Samples training data proportionally to gradient magnitude or variance contribution to improve stochastic optimization efficiency, demonstrating that not all samples contribute equally to learning.

[📄 PDF Anotado](annotated_papers/(2019)%20not%20all%20samples%20are%20created%20equal.pdf)

---

## 2018 <

### Online Batch Selection for Faster Training of Neural Networks (2016)
Prioritizes high-loss samples dynamically during training. Demonstrates improved convergence speed in deep networks through intelligent batch selection strategies.

[📄 PDF Anotado](annotated_papers/(2016)%20online%20batch%20selection%20for%20faster%20training%20of.pdf)

### Curriculum Learning (Bengio et al., 2009)
Introduces structured training progression from easy to hard samples. Shows faster convergence and improved generalization by organizing examples from easier to harder, improving optimization stability.

[📄 PDF Anotado](annotated_papers/(2009)%20curriculum%20learning.pdf)

---
