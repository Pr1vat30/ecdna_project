import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class eccDNAEvaluator:
    """
    Computes embedding space evaluation metrics (Recall@K, mAP) 
    and handles publication-quality dimensionality reduction visualization.
    """
    def __init__(self, model, config, label_encoder):
        self.model = model
        self.config = config
        self.label_encoder = label_encoder
        self.device = config.device
        
    def extract_embeddings(self, loader):
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                embeds = self.model.get_embedding(features)
                all_embeddings.append(embeds.cpu().numpy())
                all_labels.append(labels.numpy())
                
        return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

    def evaluate_retrieval(self, embeddings, labels):
        # Calculate leave-one-out query search performance
        N, d = embeddings.shape
        
        # Calculate pairwise cosine similarity matrix: [N, N]
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Mask out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, -np.inf)
        
        # Sort indices: descending similarity
        sorted_indices = np.argsort(-sim_matrix, axis=1)
        
        recall_1 = 0
        recall_5 = 0
        ap_sum = 0.0
        
        for i in range(N):
            query_label = labels[i]
            # Exclude self-similarity by dropping the last ranked item (which is self-match set to -inf)
            retrieved_labels = labels[sorted_indices[i][:-1]]
            
            # Recall@1
            if retrieved_labels[0] == query_label:
                recall_1 += 1
            # Recall@5
            if query_label in retrieved_labels[:5]:
                recall_5 += 1
                
            # AP calculation
            correct_positions = (retrieved_labels == query_label)
            num_relevant = np.sum(correct_positions)
            if num_relevant == 0:
                continue
                
            ranks = np.where(correct_positions)[0] + 1  # 1-indexed ranks
            precision_at_ranks = np.arange(1, num_relevant + 1) / ranks
            ap_sum += np.sum(precision_at_ranks) / num_relevant
            
        return {
            "Recall@1": recall_1 / N,
            "Recall@5": recall_5 / N,
            "mAP": ap_sum / N
        }
        
    def plot_embeddings(self, embeddings, labels, save_path):
        # Reduce dimensions to 2D
        pca = PCA(n_components=2, random_state=self.config.seed)
        emb_pca = pca.fit_transform(embeddings)
        
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.config.seed)
        emb_tsne = tsne.fit_transform(embeddings)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Decode disease names
        labels_decoded = self.label_encoder.inverse_transform(labels)
        unique_labels = np.unique(labels_decoded)
        
        # Elegant publication color palette
        palette = sns.color_palette("husl", len(unique_labels))
        
        # PCA Plot
        sns.scatterplot(
            x=emb_pca[:, 0], y=emb_pca[:, 1], hue=labels_decoded,
            palette=palette, alpha=0.6, ax=axes[0], s=20
        )
        axes[0].set_title("PCA Projection of eccDNA Embeddings", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("PC 1", fontsize=11)
        axes[0].set_ylabel("PC 2", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.4)
        
        # t-SNE Plot
        sns.scatterplot(
            x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=labels_decoded,
            palette=palette, alpha=0.6, ax=axes[1], s=20
        )
        axes[1].set_title("t-SNE Projection of eccDNA Embeddings", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("t-SNE 1", fontsize=11)
        axes[1].set_ylabel("t-SNE 2", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        # Format Legends
        for ax in axes:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
            
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
