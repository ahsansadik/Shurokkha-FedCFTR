import torch
from model import clone_state_dict


class FedAvgAggregator:
    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_states, sample_counts):
        total_samples = float(sum(sample_counts))
        merged = {}
        for key in client_states[0].keys():
            acc = None
            for state, count in zip(client_states, sample_counts):
                weight = count / total_samples
                term = state[key].float() * weight
                acc = term if acc is None else acc + term
            orig_dtype = client_states[0][key].dtype
            if orig_dtype.is_floating_point:
                merged[key] = acc.to(orig_dtype)
            else:
                merged[key] = torch.round(acc).to(orig_dtype)
        self.global_model.load_state_dict(clone_state_dict(merged, self.device))
        return clone_state_dict(self.global_model.state_dict(), "cpu")

    def get_global_weights(self):
        return clone_state_dict(self.global_model.state_dict(), "cpu")
