import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, DEVICE, LEARNING_RATE, LOCAL_EPOCHS
from model import clone_state_dict


class FederatedClientNode:
    def __init__(self, node_name, model, silo_data):
        self.node_name = node_name
        self.model = model.to(DEVICE)
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(silo_data["X_train"], dtype=torch.float32),
                torch.tensor(silo_data["y_train"], dtype=torch.long),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=len(silo_data["X_train"]) > BATCH_SIZE,
        )
        self.test_loader = DataLoader(
            TensorDataset(
                torch.tensor(silo_data["X_test"], dtype=torch.float32),
                torch.tensor(silo_data["y_test"], dtype=torch.long),
            ),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
        )
        self.sample_count = len(silo_data["X_train"])

    def receive_global_weights(self, global_state):
        self.model.load_state_dict(clone_state_dict(global_state, DEVICE))

    def train_local(self, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for bx, by in self.train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
        avg_loss = total_loss / max(1, steps)
        return clone_state_dict(self.model.state_dict(), "cpu"), self.sample_count, avg_loss
