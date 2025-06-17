# Continual Learning for Circuit Netlist Graphs using Graph Neural Networks

This repository contains code and experiments for our research on applying continual (lifelong) learning methods to gate-level circuit netlists modeled as graphs. We explore how state-of-the-art continual learning (CL) algorithms help mitigate catastrophic forgetting in GNNs across various EDA-relevant tasks such as node classification, graph classification, and link prediction.

## 🧠 Motivation

Most graph learning models for EDA assume static datasets and struggle to generalize when circuit topologies evolve or new gate types are introduced. In real-world workflows, circuits are developed iteratively, requiring models that can learn continually without forgetting previous tasks.

This project introduces a benchmark and interpretable continual learning framework specifically for gate-level netlists parsed from standard EDA benchmarks (ISCAS85 and EPFL).


## 🧪 Tasks Covered

- **Node Classification:** Predict gate type for each node (e.g., AND, OR).
- **Graph Classification:** Classify netlist graphs by functional type (e.g., control vs arithmetic).
- **Link Prediction:** Predict whether a wire exists between two gates.

Each task is designed under a **continual learning setting** with 4 binary tasks constructed over disjoint gate-type pairs.

## 🔁 Continual Learning Algorithms

The following CL algorithms are implemented using a fixed GNN backbone (Graph Convolutional Network):

### 🔒 Parameter Regularization
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)

### 🔁 Replay-Based
- iCaRL (Incremental Classifier and Representation Learning)
- A-GEM (Average Gradient Episodic Memory)

### ⚡ Hybrid
- Meta-Experience Replay (MER)
- CoPE (Contrastive Continual Pretraining)

## 🔍 Explainability

We provide parameter-level explainability using:
- Parameter drift analysis
- Fisher and SI importance-weighted changes
- Gradient conflict (A-GEM)
- Embedding drift (iCaRL, MER)

These metrics are correlated with accuracy drops to diagnose forgetting.

## 📦 Setup

```bash
# Create environment
conda create -n circuit-cl python=3.9
conda activate circuit-cl

# Install dependencies
- DGL
- Pytorch
- NetworkX


