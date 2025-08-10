# Language Modeling with LSTM and AWD-LSTM on WikiText-2

This repository implements language modeling using **LSTM** and **AWD-LSTM** (ASGD Weight-Dropped LSTM) architectures, trained on the **WikiText-2** dataset.  
The goal is to compare a standard LSTM with an improved AWD-LSTM that applies advanced regularization and optimization strategies from the paper *"Regularizing and Optimizing LSTM Language Models"* by Stephen Merity et al.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Models](#models)  
   - [Base LSTM](#base-lstm)  
   - [AWD-LSTM](#awd-lstm)  
4. [Results](#results)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [References](#references)  

---

## Introduction
Language modeling is a key NLP task: predicting the next word given preceding context.  
This project implements and evaluates:

- **Base Model:** Two-layer LSTM language model  
- **Improved Model:** AWD-LSTM with weight dropping, variational dropout, weight tying, and ASGD for better performance and generalization  

---

## Dataset
The **WikiText-2** dataset contains high-quality Wikipedia articles with ~33,000 unique words.  
It is well-suited for language modeling due to its rich linguistic structure and long-range dependencies.

- **Training:** ~2M tokens  
- **Validation:** ~200K tokens  
- **Test:** ~200K tokens  

---

## Models

### Base LSTM
- Two-layer LSTM network  
- Fully connected output layer for next-word prediction  
- Trained with cross-entropy loss  
- Evaluated using **perplexity**  

### AWD-LSTM
Enhancements over the base LSTM:
- **Weight Dropping:** Dropout on recurrent weights  
- **ASGD:** Averaged Stochastic Gradient Descent  
- **Variational Dropout:** Same dropout mask across time steps  
- **Weight Tying:** Shared weights between embedding and output layers  

---

## Results

| Model        | Validation Perplexity | Test Perplexity |
|--------------|-----------------------|-----------------|
| LSTM (Base)  | 128.3                 | 121.7           |
| AWD-LSTM     | **80.27**              | **77.11**       |

The AWD-LSTM substantially outperforms the base LSTM, confirming the benefits of advanced regularization.

---

## Installation
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
