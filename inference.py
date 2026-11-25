import os
import numpy as np
import torch
from config import CHECKPOINT_DIR, DEVICE
from model import CFTRMarkerNet


def predict_cftr_compound(features, input_dim, num_classes=2, checkpoint_name="shurokkha_fedcftr_global.pt"):
    ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    model = CFTRMarkerNet(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    tensor_x = torch.tensor(np.asarray(features, dtype=np.float32), device=DEVICE)
    if tensor_x.ndim == 1:
        tensor_x = tensor_x.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds.tolist(), probs.tolist()


if __name__ == "__main__":
    sample = [[475.49, 0.0, 1.51, 0.0], [269.31, 0.0, 2.36, 0.0]]
    labels, probabilities = predict_cftr_compound(sample, input_dim=4)
    print({"predictions": labels, "probabilities": probabilities})
