import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class eccDNADataset(Dataset):
    """
    Standard PyTorch dataset representing the numerical features and disease labels
    of the preprocessed eccDNA sequences.
    """
    def __init__(self, features, labels):
        if torch.is_tensor(features):
            self.features = features.clone().detach().to(dtype=torch.float32)
        else:
            self.features = torch.tensor(features, dtype=torch.float32)
            
        if torch.is_tensor(labels):
            self.labels = labels.clone().detach().to(dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PKBatchSampler(Sampler):
    """
    Custom PyTorch sampler that yields batches containing P classes and K samples per class.
    Ensures stable batch gradient dynamics for metric and contrastive learning loss.
    """
    def __init__(self, labels, P, K, num_batches=None):
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        
        # Group indices by class label
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        if len(self.classes) < P:
            raise ValueError(f"Number of classes ({len(self.classes)}) is smaller than P ({P})")
            
        self.num_batches = num_batches if num_batches is not None else len(self.labels) // (P * K)
        
    def __iter__(self):
        for _ in range(self.num_batches):
            sampled_classes = np.random.choice(self.classes, size=self.P, replace=False)
            batch = []
            for c in sampled_classes:
                indices = self.class_to_indices[c]
                replace = len(indices) < self.K
                sampled_indices = np.random.choice(indices, size=self.K, replace=replace)
                batch.extend(sampled_indices)
            yield batch
            
    def __len__(self):
        return self.num_batches

class DatasetBuilder:
    """
    Constructs the train, validation, and test datasets.
    Handles label encoding, stratified splits, and scaling.
    To prevent data leakage, the scaler is fit ONLY on the training split.
    """
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def build_datasets(self):
        # Load clean dataset, specifying low_memory=False for mixed types in metadata columns
        df = pd.read_csv(self.config.dataset_path, low_memory=False)
        
        X = df[self.config.feature_cols].values.astype(np.float32)
        y = self.label_encoder.fit_transform(df[self.config.label_col].values)
        
        # Stratified splits
        indices = np.arange(len(y))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices, y, test_size=(self.config.test_size + self.config.val_size),
            stratify=y, random_state=self.config.seed
        )
        
        val_prop_of_temp = self.config.val_size / (self.config.test_size + self.config.val_size)
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp, test_size=(1.0 - val_prop_of_temp),
            stratify=y_temp, random_state=self.config.seed
        )
        
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        X_test, y_test = X[idx_test], y[idx_test]
        
        # Fit scaler ONLY on training features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        train_dataset = eccDNADataset(X_train, y_train)
        val_dataset = eccDNADataset(X_val, y_val)
        test_dataset = eccDNADataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset
