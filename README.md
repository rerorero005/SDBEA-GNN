# SDBEA-GNN: A Smile-enhanced Dual-Branch Embedding-Aware Graph Neural Networks With Adaptive Cross-View Fusion for Drug Repurposing

## Overview
This project implements a dual-branch graph neural network for drug–disease association prediction. The model integrates multimodal drug representations by fusing pre-trained semantic embeddings with RDKit molecular fingerprints. It employs a dual-channel architecture combining:
- Topology encoder based on Graph Convolutional Matrix Completion for relation-aware message passing on heterogeneous drug–disease association graphs
- Similarity encoder (FGCN) for feature propagation over KNN similarity graphs
- Gated re-fusion mechanism that adaptively calibrates structural similarity features with SMILES-derived molecular fingerprints
- Attention-based fusion that dynamically integrates the two complementary branches

---
![Main Model Architecture](image/main.png)
---

## 🔗 Pretrained Resources Used for Feature Initialisation

| Resource | Purpose in DREAM-GNN | Link |
| -------- | ------------------- | ---- |
| **ChemBERTa (PubChem10M, BPE 450 k)** | 1024-dim SMILES embeddings for small-molecule drugs | [`seyonec/PubChem10M_SMILES_BPE_450k`](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k)|
| **ESM-2 (650 M, UR50D)** | 1280-dim protein sequence embeddings for biologics | [`facebookresearch/esm2_t33_650M_UR50D`](https://huggingface.co/facebookresearch/esm2_t33_650M_UR50D)|
| **BioBERT (v1.1 base-cased)** | 768-dim biomedical text embeddings for disease terms | [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)|
| **DrugBank** | Curated drug metadata & identifiers | [DrugBank Online](https://go.drugbank.com/)|
| **OMIM** | Curated disease phenotype information | [omim.org](https://www.ncbi.nlm.nih.gov/omim)|

---


## Files Description

- `data_loader.py`: Handles data loading, preprocessing, and cross-validation splits
- `model.py`: Defines the neural network architecture
- `layers.py`: Contains custom layer implementations (GCMC, GCN, Attention, Decoder)
- `train.py`: Main training script with seed-based experiments
- `ablation.py`: Ablation study script for hyperparameter analysis
- `evaluation.py`: Model evaluation metrics (AUROC, AUPR)
- `augmentation.py`: Graph data augmentation techniques
- `utils.py`: Utility functions for graph processing and logging

## Usage

### Basic Training

Run training with default parameters:

```bash
python train.py --data_name lrssl --device 0
```

## Model Architecture

1. **GCMC Module**: Processes drug-disease interaction graph with relation-specific transformations
2. **FGCN Module**: Processes drug and disease similarity graphs separately
3. **Attention Fusion**: Combines topology and feature representations
4. **MLP Decoder**: Predicts association scores

## Data Format

Input data should be in MATLAB (.mat) format containing:
- `didr`: Drug-disease association matrix
- `drug`: Drug similarity matrix
- `disease`: Disease similarity matrix
- `drug_embed`: Drug feature embeddings
- `disease_embed`: Disease feature embeddings
- `Wrname`: Drug identifiers
