import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter

from fraud_detection.data.elliptic import EllipticDataset
from fraud_detection.models.gcn import GCN

models_map = {"gcn": GCN}
datasets_map = {"elliptic": EllipticDataset}


class Trainer:
    def __init__(self, config):
        self.config = config

        # --- Device safety check ---
        requested_device = self.config.train.device

        if requested_device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            self.config.train.device = "cpu"

        self.device = torch.device(self.config.train.device)

        # Dataset
        dataset_obj = datasets_map[self.config.train.dataset](config.dataset)
        self.dataset = dataset_obj.pyg_dataset().to(self.device)

        # Model input dim from dataset
        self.config.model.input_dim = self.dataset.num_node_features

        # Model (float32)
        self.model = models_map[self.config.train.model](config.model).to(self.device)

        # Loss for logits (Option B)
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

        # TensorBoard logging location
        run_dir = getattr(self.config.train, "run_dir", "artifacts/runs")
        os.makedirs(run_dir, exist_ok=True)
        self.tensorboard = SummaryWriter(log_dir=run_dir)

        self.metrics_outputs = {
            "train": {
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            },
            "eval": {
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            },
        }

    def compute_metrics(self, preds, labels, mode, threshold=0.5):
        preds = preds > threshold

        accuracy = accuracy_score(labels, preds)
        f1_micro = f1_score(labels, preds, average="micro")
        f1_macro = f1_score(labels, preds, average="macro")
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=1)
        cm = confusion_matrix(labels, preds)

        self.metrics_outputs[mode]["accuracy"].append(accuracy)
        self.metrics_outputs[mode]["f1_micro"].append(f1_micro)
        self.metrics_outputs[mode]["f1_macro"].append(f1_macro)
        self.metrics_outputs[mode]["recall"].append(recall)
        self.metrics_outputs[mode]["precision"].append(precision)
        self.metrics_outputs[mode]["confusion_matrix"].append(cm)

        return {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "recall": recall,
            "precision": precision,
        }

    def train(self):
        for epoch in range(1, self.config.train.num_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(self.dataset)  # [N]
            loss = self.criterion(
                logits[self.dataset.train_idx],
                self.dataset.y[self.dataset.train_idx],
            )

            loss.backward()
            self.optimizer.step()

            probs = torch.sigmoid(logits)

            # Metrics: train
            labels = self.dataset.y.detach().cpu().numpy()[self.dataset.train_idx]
            preds = probs.detach().cpu().numpy()[self.dataset.train_idx]
            train_results = self.compute_metrics(
                preds, labels, mode="train", threshold=0.5
            )

            # Metrics: eval
            self.model.eval()
            labels = self.dataset.y.detach().cpu().numpy()[self.dataset.valid_idx]
            preds = probs.detach().cpu().numpy()[self.dataset.valid_idx]
            eval_results = self.compute_metrics(
                preds, labels, mode="eval", threshold=0.5
            )

            # Scheduler step (uses loss)
            self.scheduler.step(loss.item())

            if not epoch % self.config.train.print_freq:
                print(
                    f"epoch: {epoch}:\n"
                    f"loss: {loss.item():.4f}\n"
                    f"Train results: {train_results}\n"
                    f"Evaluation results: {eval_results}"
                )

            # TensorBoard
            self.tensorboard.add_scalar("train/loss", loss.item(), epoch)
            for metric, value in train_results.items():
                self.tensorboard.add_scalar(f"train/{metric}", value, epoch)
            for metric, value in eval_results.items():
                self.tensorboard.add_scalar(f"eval/{metric}", value, epoch)

    def test(self, dataset=None, labeled_only=False, threshold=0.5):
        dataset = dataset or self.dataset

        self.model.eval()
        logits = self.model(dataset)  # [N]
        probs = torch.sigmoid(logits)  # [N]

        if labeled_only:
            preds = probs.detach().cpu().numpy()
        else:
            preds = probs.detach().cpu().numpy()[dataset.test_idx]

        pred_labels = preds > threshold
        return preds, pred_labels

    def save(self, file_name):
        file_name = f"{file_name}.pt" if not file_name.endswith(".pt") else file_name
        save_dir = getattr(self.config.train, "save_dir", "artifacts/weights")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved weights to `{save_path}`")

    def visualize(self, dataset: EllipticDataset, time_step: int, save_to=None):
        pred_scores, pred_labels = self.test(
            dataset.pyg_dataset().to(self.device),
            labeled_only=True,
        )

        # Use the helper for clarity (your EllipticDataset already has it)
        node_list = dataset.nodes_at_time(time_step)

        edge_tuples = []
        edge_index = dataset.edge_index.t().cpu().numpy()  # [E, 2]
        node_set = set(node_list)
        for u, v in edge_index:
            if (u in node_set) or (v in node_set):
                edge_tuples.append((int(u), int(v)))

        node_color = []
        for node_id in node_list:
            if node_id in dataset.illicit_ids:
                label = "red"
            elif node_id in dataset.licit_ids:
                label = "green"
            else:
                label = "orange" if pred_labels[node_id] else "blue"
            node_color.append(label)

        G = nx.Graph()
        G.add_edges_from(edge_tuples)

        plt.figure(figsize=(16, 16))
        plt.title(f"Time period: {time_step}")
        nx.draw_networkx(
            G,
            nodelist=node_list,
            node_color=node_color,
            node_size=6,
            with_labels=False,
        )

        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            plt.savefig(save_to)
            print(f"Graph visualization saved to `{save_to}`")
