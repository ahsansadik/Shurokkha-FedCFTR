import torch
from torch.utils.data import DataLoader, TensorDataset
from config import DEVICE


def evaluate_model(model, X_test, y_test, batch_size=64):
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            preds = model(bx).argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)
            tp += ((preds == 1) & (by == 1)).sum().item()
            fp += ((preds == 1) & (by == 0)).sum().item()
            fn += ((preds == 0) & (by == 1)).sum().item()
    acc = 100.0 * correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": round(acc, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total": total,
    }
