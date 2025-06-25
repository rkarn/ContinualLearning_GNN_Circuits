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
- Meta-Experience Replay (MER): Combines experience replay with meta-gradient optimization.
- CoPE (Contrastive Continual Pretraining): Combines continual learning with self-supervised pretraining.

## 🔍 Explainability

We provide parameter-level explainability using:
- Parameter drift analysis
- Fisher and SI importance-weighted changes
- Gradient conflict (A-GEM)
- Embedding drift (iCaRL, MER)

These metrics are correlated with accuracy drops to diagnose forgetting.


The value of key-hyperparameters are:

### Table: Hyperparameters shared across CL methods

| **Hyperparameter**                | **Methods**                                     | **Value**                   |
|----------------------------------|--------------------------------------------------|-----------------------------|
| Hidden dimension                 | EWC, SI, iCaRL, A-GEM, MER, CoPE                | 64                          |
| Epochs per task (fine-tune)      | EWC, SI, iCaRL, A-GEM, MER                      | 50                          |
| Memory buffer size per task      | A-GEM, MER                                      | 200 graphs                  |
| Optimizer                        | EWC, SI                                         | Adam (lr=1e-2)              |
|                                  | iCaRL, A-GEM                                    | Adam (lr=1e-3)              |
|                                  | MER                                             | SGD (meta lr=1e-2)          |
|                                  | CoPE                                            | Adam (pre:5e-4; cls:5e-3)   |
| Regularization strength (λ)      | EWC, SI                                         | 1000.0                      |
| Temperature (T)                  | iCaRL                                           | 2.0                         |
| Distillation weight (β)          | iCaRL                                           | 1.0                         |
| Exemplars per class              | iCaRL                                           | 20 nodes                    |
| Memory batch size                | A-GEM                                           | 100 graphs                  |
| Inner-loop learning rate (α)     | MER                                             | 0.005                       |
| Meta-learning rate               | MER                                             | 0.01                        |
| MER batch size                   | MER                                             | 64                          |
| Dropout (in CoPE encoder)        | CoPE                                            | 0.4                         |
| Mask ratio (feature drop)        | CoPE                                            | 0.2                         |
| Pretext epochs                   | CoPE                                            | 30                          |
| Classifier epochs                | CoPE                                            | 40                          |



## 📦 Setup

```bash
# Create environment
conda create -n circuit-cl python=3.9
conda activate circuit-cl

# Install dependencies
- DGL
- Pytorch
- NetworkX


