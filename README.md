# Shurokkha FedCFTR

A modular PyTorch Federated Learning framework for classifying Cystic Fibrosis (`CFTR`) bioactivity markers across non-IID clinical and laboratory sub-systems (`Hospital_A`, `Hospital_B`, `Lab_C`) using sample-weighted Federated Averaging (`FedAvg`) while keeping raw records strictly local.

## Repository Structure
- `config.py` — Runtime device selection, client silo names (`Hospital_A`, `Hospital_B`, `Lab_C`), training hyperparameters, feature columns, and paths
- `data_loader.py` — Loads CSV/XLSX `CFTR` bioactivity tables, cleans/encodes features, and generates a fallback `cftr_bioactivity.csv` dataset if none is present
- `partitioner.py` — Splits data into non-IID institutional silos and applies train-split-only feature standardization per client
- `model.py` — Defines `CFTRMarkerNet` and weight state-dictionary cloning helpers
- `client_node.py` — Encapsulates local training loops and weight extraction for each sub-system
- `aggregator.py` — Implements sample-weighted `FedAvg` parameter aggregation and global model synchronization
- `evaluator.py` — Computes held-out test accuracy, precision, recall, and F1-score
- `privacy_guard.py` — Validates weight tensor keys and sizes, computes SHA-256 weight fingerprints, and emits zero-raw-data audit records
- `train_federated.py` — Runs the multi-round federated training pipeline and saves `checkpoints/shurokkha_fedcftr_global.pt`
- `inference.py` — Loads a saved checkpoint and returns predicted class labels and class probabilities for input feature rows

## Setup & Usage
1. Install dependencies:
   ```bash
   pip install torch numpy pandas openpyxl
