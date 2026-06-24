import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseEncoder(nn.Module):
    """
    Encoder network that takes multi-feature eccDNA vectors and maps them 
    into a low-dimensional embedding space, applying L2 normalization to 
    constrain the output onto the unit hypersphere.
    """
    def __init__(self, input_dim, hidden_dims, embed_dim, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        raw_embed = self.encoder(x)
        # Apply L2 Normalization onto the unit hypersphere
        normalized_embed = F.normalize(raw_embed, p=2, dim=1)
        return normalized_embed

class CNNSiameseEncoder(nn.Module):
    """
    1D Convolutional Neural Network (1D-CNN) Siamese Encoder.
    Treats the input feature vector as a spatial/sequential signal.
    """
    def __init__(self, input_dim, embed_dim, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate flattened dimension dynamically
        self.flat_dim = 64 * ((input_dim // 2) // 2)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )
        
    def forward(self, x):
        # Shape: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        raw_embed = self.fc(x)
        return F.normalize(raw_embed, p=2, dim=1)

class SoftmaxClassifier(nn.Module):
    """
    Classification head mapping from the embedding representation space
    to the target number of disease classes.
    """
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class eccDNAModel(nn.Module):
    """
    Unified multi-task model coupling representation learning (SiameseEncoder)
    with disease classification (SoftmaxClassifier).
    """
    def __init__(self, config, num_classes):
        super().__init__()
        model_type = getattr(config, 'model_type', 'mlp')
        if model_type == 'cnn':
            self.encoder = CNNSiameseEncoder(
                input_dim=config.input_dim,
                embed_dim=config.embed_dim,
                dropout=config.dropout
            )
        else:
            self.encoder = SiameseEncoder(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                embed_dim=config.embed_dim,
                dropout=config.dropout
            )
            
        self.classifier = SoftmaxClassifier(
            embed_dim=config.embed_dim,
            hidden_dim=config.classifier_hidden,
            num_classes=num_classes
        )
        
    def forward(self, x):
        embeds = self.encoder(x)
        logits = self.classifier(embeds)
        return embeds, logits
        
    def get_embedding(self, x):
        return self.encoder(x)
