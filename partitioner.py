import numpy as np
from config import CLIENT_NAMES, RANDOM_SEED


def split_non_iid_silos(X, y, client_names=CLIENT_NAMES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    num_clients = len(client_names)
    indices = np.arange(len(X))
    sorted_order = indices[np.argsort(y)]
    shards = np.array_split(sorted_order, num_clients * 2)
    shard_ids = rng.permutation(len(shards))
    silos = {}
    for idx, name in enumerate(client_names):
        chosen = np.concatenate([shards[shard_ids[idx * 2]], shards[shard_ids[idx * 2 + 1]]])
        rng.shuffle(chosen)
        split_at = max(1, int(0.8 * len(chosen)))
        train_idx = chosen[:split_at]
        test_idx = chosen[split_at:] if split_at < len(chosen) else chosen
        silos[name] = {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
        }
    return silos


def standardize_silo(silo):
    mean = silo["X_train"].mean(axis=0, keepdims=True)
    std = silo["X_train"].std(axis=0, keepdims=True) + 1e-8
    return {
        "X_train": (silo["X_train"] - mean) / std,
        "y_train": silo["y_train"],
        "X_test": (silo["X_test"] - mean) / std,
        "y_test": silo["y_test"],
    }
