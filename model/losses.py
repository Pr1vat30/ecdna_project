import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon).
    Pulls positive pairs (same disease) together and pushes negative pairs apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Number of labels does not match number of embeddings')
            
        # Create mask: mask[i, j] = 1 if labels[i] == labels[j]
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Calculate cosine similarity matrix (embeddings are already L2 normalized)
        anchor_dot_contrast = torch.div(
            torch.matmul(embeddings, embeddings.T),
            self.temperature
        )
        
        # For numerical stability: subtract max value
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast (diagonal elements)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive columns
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss is negative average
        loss = -mean_log_prob_pos.mean()
        return loss

class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss.
    Mines the hardest positive and hardest negative sample for each anchor in the batch.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device
        
        # Pairwise distance matrix computation (L2 distance squared)
        dot_product = torch.matmul(embeddings, embeddings.T)
        distances = torch.clamp(2.0 - 2.0 * dot_product, min=0.0)
        
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        
        # Hardest positive: maximum distance among same-class items
        pos_distances = distances * mask_pos
        hardest_pos, _ = torch.max(pos_distances, dim=1)
        
        # Hardest negative: minimum distance among different-class items
        max_dist = distances.max().item()
        neg_distances = distances + (mask_pos * (max_dist + 1e5))
        hardest_neg, _ = torch.min(neg_distances, dim=1)
        
        # Margin loss
        loss = torch.clamp(hardest_pos - hardest_neg + self.margin, min=0.0)
        return loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Combines a metric learning loss (SupCon / Triplet) with a classification loss (Cross-Entropy).
    """
    def __init__(self, metric_loss_fn, classification_loss_fn):
        super().__init__()
        self.metric_loss = metric_loss_fn
        self.cls_loss = classification_loss_fn
        
    def forward(self, embeddings, logits, labels, lam):
        m_loss = self.metric_loss(embeddings, labels)
        c_loss = self.cls_loss(logits, labels)
        total_loss = lam * m_loss + (1.0 - lam) * c_loss
        return total_loss, m_loss, c_loss
