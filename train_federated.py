import os
import numpy as np
import torch
from aggregator import FedAvgAggregator
from client_node import FederatedClientNode
from config import CHECKPOINT_DIR, CLIENT_NAMES, DEVICE, ROUNDS
from data_loader import extract_features_and_labels, load_cftr_dataframe
from evaluator import evaluate_model
from model import CFTRMarkerNet
from partitioner import split_non_iid_silos, standardize_silo
from privacy_guard import build_audit_record, verify_weight_payload


def run_federated_pipeline():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    df = load_cftr_dataframe()
    X, y, label_map = extract_features_and_labels(df)
    silos = split_non_iid_silos(X, y, CLIENT_NAMES)
    clients = []
    global_X_test = []
    global_y_test = []
    for name in CLIENT_NAMES:
        norm_silo = standardize_silo(silos[name])
        local_model = CFTRMarkerNet(input_dim=X.shape[1], num_classes=len(label_map))
        clients.append(FederatedClientNode(name, local_model, norm_silo))
        global_X_test.append(norm_silo["X_test"])
        global_y_test.append(norm_silo["y_test"])
    X_eval = np.vstack(global_X_test)
    y_eval = np.concatenate(global_y_test)
    boss_net = CFTRMarkerNet(input_dim=X.shape[1], num_classes=len(label_map))
    aggregator = FedAvgAggregator(boss_net, DEVICE)
    allowed_keys = list(boss_net.state_dict().keys())
    history = []
    for rnd in range(1, ROUNDS + 1):
        global_weights = aggregator.get_global_weights()
        updates = []
        counts = []
        for client in clients:
            client.receive_global_weights(global_weights)
            w_local, n_samples, loss = client.train_local()
            verify_weight_payload(w_local, allowed_keys)
            print(build_audit_record(rnd, client.node_name, n_samples, w_local))
            updates.append(w_local)
            counts.append(n_samples)
        merged_weights = aggregator.aggregate(updates, counts)
        for client in clients:
            client.receive_global_weights(merged_weights)
        metrics = evaluate_model(aggregator.global_model, X_eval, y_eval)
        history.append({"round": rnd, **metrics})
        print(f"Round {rnd:02d}/{ROUNDS:02d} | Accuracy: {metrics['accuracy']}% | F1: {metrics['f1']}")
    torch.save(aggregator.get_global_weights(), os.path.join(CHECKPOINT_DIR, "shurokkha_fedcftr_global.pt"))
    return history


if __name__ == "__main__":
    run_federated_pipeline()
