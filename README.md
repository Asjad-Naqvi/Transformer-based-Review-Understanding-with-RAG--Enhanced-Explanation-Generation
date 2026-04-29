# Transformer-based-Review-Understanding-with-RAG--Enhanced-Explanation-Generation
build a three-stage NLP system that first extracts structured information from text, retrieves relevant examples from the dataset, and then generates grounded natural language explanations. The three stages — encoding, retrieval, and generation — are implemented in Parts A, B, and C respectively, and must work together as a coherent pipeline.



# NLP Assignment 3: Transformer-based Review Understanding with RAG

**Student ID:** i23XXXX  
**Course:** CS-4063 — NLP  
**Due Date:** 29-04-26, 11:59 PM

---

## 📋 Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for Amazon product review analysis, built entirely from scratch using PyTorch without any high-level Transformer APIs or pretrained models.

### System Architecture

| Stage | Component | Description |
|-------|-----------|-------------|
| **A** | Encoder-only Transformer | Multi-task model for sentiment + helpfulness classification |
| **B** | Retrieval Module | Cosine similarity k-NN over encoder embeddings |
| **C** | Decoder-only Transformer | Autoregressive explanation generation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, falls back to CPU)
- 4 GB+ RAM

### Installation

```bash
# Clone or download the notebook
# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn

### Data:

https://nijianmo.github.io/amazon/index.html




