import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 0. UTILS
# ==========================================

def dict_to_device(batch_dict, device):
    """Sposta un dizionario di tensori sul device specificato (CPU/GPU)"""
    return {k: v.to(device) for k, v in batch_dict.items()}

# ==========================================
# 1. DATASET
# ==========================================

class DatasetBuilder:
    def __init__(self, path, min_samples=1000, test_size=0.3, val_size=0.5, random_state=42):
        self.path = path
        self.min_samples = min_samples
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.X = {}  # Dizionario per le feature
        self.y = None
        self.X_train, self.X_val, self.X_test = {}, {}, {}
        self.y_train, self.y_val, self.y_test = None, None, None

        self.le = LabelEncoder()
        self.input_dims = {}
        self.num_class = None
        self.weights = None
        self.df = None

    def load_and_filter(self):
        self.df = pd.read_csv(self.path)
        class_counts = self.df['Disease'].value_counts()
        valid_classes = class_counts[class_counts >= self.min_samples].index
        self.df = self.df[self.df['Disease'].isin(valid_classes)].reset_index(drop=True)

    def build_dataset(self):
        # 1. Definiamo le liste delle colonne
        mi_cols = [f"MI_tau_{i}" for i in range(1, 101)]
        resolved_cols = [f"MI_Resolved_{base}_{tau}" for base in ['A', 'C', 'G', 'T'] for tau in range(1, 6)]
        ami_cols = ["AMI_1_20", "AMI_21_50", "AMI_51_100"]

        # Concateniamo tutti i nomi delle feature che vogliamo usare
        all_features = mi_cols + resolved_cols + ami_cols

        # 2. Estraiamo tutte le feature in un'unica matrice e le mettiamo nel dict
        # Invece di avere 123 chiavi, ne abbiamo una sola ("all_features").
        # L'MLP leggerà che la dimensione in ingresso è 123.
        self.X["all_features"] = self.df[all_features].values.astype(np.float32)
        self.input_dims["all_features"] = len(all_features)

        # 3. Encoding delle label
        self.y = self.le.fit_transform(self.df["Disease"])
        self.num_class = len(np.unique(self.y))

    def split(self):
        indices = np.arange(len(self.y))

        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices, self.y, test_size=self.test_size, stratify=self.y, random_state=self.random_state
        )
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp, test_size=self.val_size, stratify=y_temp, random_state=self.random_state
        )

        self.X_train = {k: v[idx_train] for k, v in self.X.items()}
        self.X_val = {k: v[idx_val] for k, v in self.X.items()}
        self.X_test = {k: v[idx_test] for k, v in self.X.items()}

        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def get_weights(self):
        self.weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)


class TripletBioDataset(Dataset):
    """Dataset per il training con Triplet Loss (supporta dizionari)"""

    def __init__(self, X_dict, y):
        self.X = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.long)
        self.labels = self.y.numpy()

        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }

    def __len__(self):
        return len(self.y)

    def _get_item_dict(self, index):
        return {k: v[index] for k, v in self.X.items()}

    def __getitem__(self, index):
        anchor = self._get_item_dict(index)
        anchor_label = self.y[index]

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label.item()])
        positive = self._get_item_dict(positive_index)

        negative_label = np.random.choice(list(set(self.label_to_indices.keys()) - {anchor_label.item()}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative = self._get_item_dict(negative_index)

        return anchor, positive, negative, anchor_label


class BioDataset(Dataset):
    """Dataset standard per Validation e Testing (supporta dizionari)"""

    def __init__(self, X_dict, y):
        self.X = {k: torch.tensor(v, dtype=torch.float32) for k, v in X_dict.items()}
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self._get_item_dict(idx), self.y[idx]

    def _get_item_dict(self, index):
        return {k: v[index] for k, v in self.X.items()}

# ==========================================
# 2. MODEL
# ==========================================

class ResNetSiameseBioNet(nn.Module):
    def __init__(self, input_dims_dict, d_model=256, embed_dim=128, num_classes=5, dropout_rate=0.3):
        super(ResNetSiameseBioNet, self).__init__()

        # 1. Calcola la dimensione input
        total_input_dim = sum(input_dims_dict.values())

        # 2. Proiezione Iniziale
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # 3. Blocco Residuale 1
        self.res_block1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model)
        )

        # 4. Blocco Residuale 2
        self.res_block2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model)
        )

        # 5. Layer Finale per Embedding
        self.fc_embed = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

        # 6. Classificatore Ausiliario
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward_features(self, x_dict):
        # Estrae e concatena dinamicamente
        features = [x_dict[k] for k in x_dict.keys()]
        x = torch.cat(features, dim=1)

        # Passaggio Rete
        x = self.input_proj(x)

        # Skip Connections (Residui)
        x = F.gelu(x + self.res_block1(x))
        x = F.gelu(x + self.res_block2(x))

        return self.fc_embed(x)

    def forward(self, x_dict):
        raw_embedding = self.forward_features(x_dict)
        logits = self.classifier(raw_embedding)
        metric_embedding = F.normalize(raw_embedding, p=2, dim=1)
        return metric_embedding, logits

    def get_embedding(self, x_dict):
        raw_embedding = self.forward_features(x_dict)
        return F.normalize(raw_embedding, p=2, dim=1)

# ==========================================
# 3. TRAIN & TEST LOOPS
# ==========================================

class SiameseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.triplet_loss_fn = nn.TripletMarginLoss(margin=config['triplet_margin'], p=2)

        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.ce_loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, leave=False, desc="Training Batches")

        for anchor_dict, positive_dict, negative_dict, labels in loop:
            anchor = dict_to_device(anchor_dict, self.device)
            positive = dict_to_device(positive_dict, self.device)
            negative = dict_to_device(negative_dict, self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            emb_anchor, logits = self.model(anchor)
            emb_positive = self.model.get_embedding(positive)
            emb_negative = self.model.get_embedding(negative)

            loss_triplet = self.triplet_loss_fn(emb_anchor, emb_positive, emb_negative)
            loss_ce = self.ce_loss_fn(logits, labels)

            loss = (self.alpha * loss_triplet) + (self.beta * loss_ce)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_dict, labels in self.val_loader:
                x = dict_to_device(x_dict, self.device)
                labels = labels.to(self.device)

                _, logits = self.model(x)
                loss_ce = self.ce_loss_fn(logits, labels)

                val_loss += loss_ce.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return val_loss / len(self.val_loader), accuracy

    def fit(self, epochs, scheduler=None):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if scheduler:
                scheduler.step(val_loss)  # Riduce il LR se la Val Loss non scende per 5 epoche


class SiameseTester:
    def __init__(self, model, test_loader, device, label_encoder=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.label_encoder = label_encoder

    def evaluate_classification(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_dict, labels in self.test_loader:
                x = dict_to_device(x_dict, self.device)
                _, logits = self.model(x)

                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"\n--- 1. RISULTATI CLASSIFICAZIONE (Cross-Entropy) ---")
        print(f"Accuracy Classificazione: {acc:.4f}\n")

        target_names = [str(cls) for cls in self.label_encoder.classes_] if self.label_encoder else None
        print(classification_report(all_labels, all_preds, target_names=target_names))

        return all_labels, all_preds

    def evaluate_similarity(self, num_pairs=2000):
        print(f"\n--- 2. RISULTATI SIMILARITA' (Spazio Embedding) ---")
        self.model.eval()

        all_embs = []
        all_labels = []
        with torch.no_grad():
            for x_dict, labels in self.test_loader:
                x = dict_to_device(x_dict, self.device)
                emb = self.model.get_embedding(x)
                all_embs.append(emb.cpu())
                all_labels.append(labels.cpu())

        all_embs = torch.cat(all_embs)
        all_labels = torch.cat(all_labels).numpy()

        label_to_indices = {label: np.where(all_labels == label)[0] for label in np.unique(all_labels)}
        pairs = []
        y_true = []

        num_pos = num_pairs // 2
        for _ in range(num_pos):
            c = np.random.choice(list(label_to_indices.keys()))
            if len(label_to_indices[c]) >= 2:
                idx1, idx2 = np.random.choice(label_to_indices[c], 2, replace=False)
                pairs.append((all_embs[idx1], all_embs[idx2]))
                y_true.append(1)

        num_neg = num_pairs - len(pairs)
        classes = list(label_to_indices.keys())
        for _ in range(num_neg):
            c1, c2 = np.random.choice(classes, 2, replace=False)
            idx1 = np.random.choice(label_to_indices[c1])
            idx2 = np.random.choice(label_to_indices[c2])
            pairs.append((all_embs[idx1], all_embs[idx2]))
            y_true.append(0)

        emb1 = torch.stack([p[0] for p in pairs])
        emb2 = torch.stack([p[1] for p in pairs])
        distances = F.pairwise_distance(emb1, emb2).numpy()
        y_true = np.array(y_true)

        scores = -distances
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]

        y_pred = (distances <= optimal_threshold).astype(int)
        pair_acc = accuracy_score(y_true, y_pred)

        print(f"Coppie Analizzate: {len(pairs)} ({sum(y_true)} Positive, {len(y_true) - sum(y_true)} Negative)")
        print(f"ROC AUC Score:     {roc_auc:.4f}")
        print(f"Soglia Ottimale:   {optimal_threshold:.4f} (Distanza max per essere 'Simili')")
        print(f"Pair Verification Accuracy: {pair_acc:.4f}")

        return pair_acc, roc_auc

    def visualize_embeddings(self):
        print(f"\n--- 3. VISUALIZZAZIONE EMBEDDING (t-SNE) ---")
        self.model.eval()

        all_embs = []
        all_labels = []

        with torch.no_grad():
            for x_dict, labels in self.test_loader:
                x = dict_to_device(x_dict, self.device)
                emb = self.model.get_embedding(x)
                all_embs.append(emb.cpu())
                all_labels.extend(labels.numpy())

        all_embs = torch.cat(all_embs).numpy()
        all_labels = np.array(all_labels)

        print("Calcolo di t-SNE in corso...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embs_2d = tsne.fit_transform(all_embs)

        target_names = self.label_encoder.inverse_transform(all_labels) if self.label_encoder else all_labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embs_2d[:, 0],
            y=embs_2d[:, 1],
            hue=target_names,
            palette="tab10",
            alpha=0.7,
            s=50,
            edgecolor=None
        )

        plt.title("t-SNE degli Embedding (Test Set)", fontsize=16)
        plt.xlabel("Dimensione t-SNE 1", fontsize=12)
        plt.ylabel("Dimensione t-SNE 2", fontsize=12)
        plt.legend(title="Classe di Malattia", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig("tsne_embeddings_test.png", dpi=300, bbox_inches="tight")
        print("Plot salvato con successo come 'tsne_embeddings_test.png'.")
        plt.show()

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    CONFIG = {
        "path_to_data": "./datasets/eccDNA_sequences_def.csv",
        "min_samples": 1000,
        "test_size": 0.3,
        "val_size": 0.5,
        "random_state": 42,

        "batch_size": 512,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,

        "d_model": 256,
        "embed_dim": 128,

        "alpha": 0.2,
        "beta": 0.8,
        "triplet_margin": 0.5,

        "num_test_pairs": 2000
    }

    builder = DatasetBuilder(
        path=CONFIG["path_to_data"],
        min_samples=CONFIG["min_samples"],
        test_size=CONFIG["test_size"],
        val_size=CONFIG["val_size"],
        random_state=CONFIG["random_state"]
    )

    print("Caricamento e preprocessing del dataset in corso...")
    builder.load_and_filter()
    builder.build_dataset()
    builder.split()
    builder.get_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TripletBioDataset(builder.X_train, builder.y_train)
    val_dataset = BioDataset(builder.X_val, builder.y_val)
    test_dataset = BioDataset(builder.X_test, builder.y_test)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Istanziamento del nuovo modello MLP
    model = ResNetSiameseBioNet(
        input_dims_dict=builder.input_dims,
        d_model=CONFIG["d_model"],
        embed_dim=CONFIG["embed_dim"],
        num_classes=builder.num_class
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=CONFIG,
        class_weights=builder.weights
    )

    print("\n--- Inizio Addestramento ---")
    trainer.fit(epochs=CONFIG["epochs"])

    print("\n--- Inizio Test ---")
    tester = SiameseTester(
        model=model,
        test_loader=test_loader,
        device=device,
        label_encoder=builder.le
    )

    tester.evaluate_classification()
    tester.evaluate_similarity(num_pairs=CONFIG["num_test_pairs"])
    tester.visualize_embeddings()

    print("\n--- SANITY CHECK: Random Forest ---")
    # Concatena in modo sicuro garantendo che le dimensioni combacino
    X_train_rf = np.hstack([builder.X_train[k] for k in builder.X_train.keys()])
    X_test_rf = np.hstack([builder.X_test[k] for k in builder.X_test.keys()])

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

    print("Addestramento Random Forest in corso...")
    rf.fit(X_train_rf, builder.y_train)

    rf_preds = rf.predict(X_test_rf)
    print(classification_report(builder.y_test, rf_preds, target_names=builder.le.classes_))