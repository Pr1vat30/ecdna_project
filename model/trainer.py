import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from model.losses import SupervisedContrastiveLoss, MultiTaskLoss

class SiameseTrainer:
    """
    Orchestrates the training and validation loops, updates models, 
    calculates accuracies and sub-losses, and updates the dynamic lambda scheduler.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, config, checkpoint_path="models/best_mlp_baseline.pt"):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = config.device
        self.checkpoint_path = checkpoint_path
        
        # Initialize Losses
        self.metric_loss_fn = SupervisedContrastiveLoss(temperature=config.temperature)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.multitask_loss = MultiTaskLoss(self.metric_loss_fn, self.cls_loss_fn)
        
    def get_lambda(self, epoch):
        # Exponential decay: lam(t) = lam_min + (lam_max - lam_min) * exp(-t / decay)
        return self.config.lambda_min + (self.config.lambda_max - self.config.lambda_min) * np.exp(-epoch / self.config.lambda_decay)
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_metric = 0.0
        epoch_cls = 0.0
        correct = 0
        total = 0
        
        lam = self.get_lambda(epoch)
        
        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings, logits = self.model(features)
            
            # Combined Loss
            loss, m_loss, c_loss = self.multitask_loss(embeddings, logits, labels, lam)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * features.size(0)
            epoch_metric += m_loss.item() * features.size(0)
            epoch_cls += c_loss.item() * features.size(0)
            
            # Classification performance
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        return {
            "loss": epoch_loss / total,
            "metric_loss": epoch_metric / total,
            "cls_loss": epoch_cls / total,
            "accuracy": correct / total,
            "lambda": lam
        }
        
    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        epoch_metric = 0.0
        epoch_cls = 0.0
        correct = 0
        total = 0
        
        # Validation holds lambda fixed at min weight
        lam = self.config.lambda_min
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                embeddings, logits = self.model(features)
                loss, m_loss, c_loss = self.multitask_loss(embeddings, logits, labels, lam)
                
                epoch_loss += loss.item() * features.size(0)
                epoch_metric += m_loss.item() * features.size(0)
                epoch_cls += c_loss.item() * features.size(0)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return {
            "loss": epoch_loss / total,
            "metric_loss": epoch_metric / total,
            "cls_loss": epoch_cls / total,
            "accuracy": correct / total
        }
        
    def fit(self):
        history = {
            "train_loss": [], "train_metric": [], "train_cls": [], "train_acc": [], "train_lambda": [],
            "val_loss": [], "val_metric": [], "val_cls": [], "val_acc": []
        }
        
        best_val_acc = 0.0
        
        pbar = tqdm(range(self.config.epochs), desc="Training eccDNA Model")
        for epoch in pbar:
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            history["train_loss"].append(train_metrics["loss"])
            history["train_metric"].append(train_metrics["metric_loss"])
            history["train_cls"].append(train_metrics["cls_loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_lambda"].append(train_metrics["lambda"])
            
            history["val_loss"].append(val_metrics["loss"])
            history["val_metric"].append(val_metrics["metric_loss"])
            history["val_cls"].append(val_metrics["cls_loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            
            pbar.set_postfix({
                "T-Loss": f"{train_metrics['loss']:.4f}",
                "V-Loss": f"{val_metrics['loss']:.4f}",
                "T-Acc": f"{train_metrics['accuracy']:.4f}",
                "V-Acc": f"{val_metrics['accuracy']:.4f}",
                "Lambda": f"{train_metrics['lambda']:.2f}"
            })
            
            # Checkpoint saving
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                os.makedirs("models", exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
                
        return history
