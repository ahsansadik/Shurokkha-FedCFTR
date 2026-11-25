import os
import numpy as np
import pandas as pd
from config import DATASET_PATH, FEATURE_COLUMNS, RANDOM_SEED


def generate_fallback_cftr_csv(path=DATASET_PATH, rows=1500):
    rng = np.random.default_rng(RANDOM_SEED)
    half = rows // 2
    pos = pd.DataFrame({
        "Molecular Weight": np.clip(rng.normal(485.0, 65.0, half), 180.0, 900.0),
        "#RO5 Violations": rng.choice([0, 1, 2], size=half, p=[0.55, 0.30, 0.15]),
        "AlogP": np.clip(rng.normal(4.6, 1.1, half), -1.0, 8.5),
        "Molecule Max Phase": rng.choice([0, 1, 2, 3, 4], size=half, p=[0.60, 0.15, 0.10, 0.10, 0.05]),
        "Standard Type": rng.choice(["IC50", "Inhibition", "Kd"], size=half, p=[0.5, 0.35, 0.15]),
        "cftr_active": np.ones(half, dtype=np.int64),
    })
    neg = pd.DataFrame({
        "Molecular Weight": np.clip(rng.normal(330.0, 55.0, rows - half), 150.0, 850.0),
        "#RO5 Violations": rng.choice([0, 1, 2], size=rows - half, p=[0.80, 0.15, 0.05]),
        "AlogP": np.clip(rng.normal(2.4, 0.9, rows - half), -1.5, 7.5),
        "Molecule Max Phase": rng.choice([0, 1, 2, 3, 4], size=rows - half, p=[0.85, 0.08, 0.04, 0.02, 0.01]),
        "Standard Type": rng.choice(["IC50", "Inhibition", "Kd"], size=rows - half, p=[0.3, 0.5, 0.2]),
        "cftr_active": np.zeros(rows - half, dtype=np.int64),
    })
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def load_cftr_dataframe(path=DATASET_PATH):
    if not os.path.exists(path):
        return generate_fallback_cftr_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def extract_features_and_labels(df, target_col=None):
    cleaned = df.dropna(axis=1, how="all").copy()
    if target_col is None or target_col not in cleaned.columns:
        target_col = "cftr_active" if "cftr_active" in cleaned.columns else cleaned.columns[-1]
    cleaned = cleaned.dropna(subset=[target_col]).reset_index(drop=True)
    raw_y = cleaned[target_col].astype(str)
    classes = sorted(raw_y.unique())
    mapping = {c: i for i, c in enumerate(classes)}
    y = raw_y.map(mapping).to_numpy(dtype=np.int64)
    available_feats = [c for c in FEATURE_COLUMNS if c in cleaned.columns and c != target_col]
    feat_df = cleaned[available_feats].copy() if available_feats else cleaned.drop(columns=[target_col]).copy()
    for col in feat_df.columns:
        if pd.api.types.is_numeric_dtype(feat_df[col]):
            med = feat_df[col].median()
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0 if pd.isna(med) else med)
        else:
            feat_df[col] = feat_df[col].astype(str).replace("None", "0").fillna("MISSING")
    X = pd.get_dummies(feat_df, drop_first=False).to_numpy(dtype=np.float32)
    return X, y, mapping
