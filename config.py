import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
NUM_CLIENTS = 3
CLIENT_NAMES = ["Hospital_A", "Hospital_B", "Lab_C"]
ROUNDS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.003
FEATURE_COLUMNS = ["Molecular Weight", "#RO5 Violations", "AlogP", "Molecule Max Phase"]
DATASET_PATH = "cftr_bioactivity.csv"
CHECKPOINT_DIR = "checkpoints"
