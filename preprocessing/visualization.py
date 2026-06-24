import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from preprocessing.config import PreprocessingConfig

# Attempt to import UMAP, with fallback to t-SNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logger = logging.getLogger(__name__)

class EDAVisualizer:
    """
    Step 6: Exploratory Data Analysis and Visualization
    Generates publication-quality plots: sequence distributions, MI profile curves,
    2D projections (PCA / UMAP / t-SNE), and outlier diagnostic reports.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Set premium, professional seaborn plotting style
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "figure.figsize": (12, 8),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12
        })
        self.palette = sns.color_palette("muted")

    def _ensure_dir(self, file_path: str) -> str:
        """Ensures the directory for the file path exists and returns the path."""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return file_path

    def plot_distributions(self, df_features: pd.DataFrame, output_path: str) -> None:
        """
        Plots histograms and KDE density estimations for:
        - Sequence length (log scale)
        - GC content (%)
        """
        self._ensure_dir(output_path)
        logger.info(f"Plotting feature distributions to {output_path}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Sequence Length Distribution
        sns.histplot(
            data=df_features, 
            x="sequence_length", 
            hue="disease", 
            kde=True, 
            log_scale=True, 
            ax=axes[0], 
            palette=self.palette,
            element="step",
            alpha=0.4
        )
        axes[0].set_title("Distribution of Sequence Lengths (log-scale)", pad=15)
        axes[0].set_xlabel("Sequence Length (bp)")
        axes[0].set_ylabel("Count")
        
        # 2. GC Content Distribution
        sns.histplot(
            data=df_features, 
            x="GC_pct", 
            hue="disease", 
            kde=True, 
            ax=axes[1], 
            palette=self.palette,
            element="step",
            alpha=0.4
        )
        axes[1].set_title("Distribution of GC Content (%)", pad=15)
        axes[1].set_xlabel("GC Content (%)")
        axes[1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Distribution plots saved successfully.")

    def plot_mi_profiles(self, df_features: pd.DataFrame, output_path: str) -> None:
        """
        Plots the average Mutual Information (MI) profiles across lags
        for each disease class, with shaded standard error bounds.
        """
        self._ensure_dir(output_path)
        logger.info(f"Plotting average MI profiles to {output_path}...")
        
        # Extract MI column names: MI_tau_1 to MI_tau_{max_lag}
        mi_cols = [f"MI_tau_{t}" for t in range(1, self.config.max_lag + 1)]
        
        # Verify columns exist
        available_mi_cols = [c for c in mi_cols if c in df_features.columns]
        if not available_mi_cols:
            logger.warning("No MI profile columns found in DataFrame. Skipping MI profile plot.")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Group by disease and calculate mean and SEM
        grouped = df_features.groupby("disease")
        
        lags = np.arange(1, len(available_mi_cols) + 1)
        
        for idx, (disease, group) in enumerate(grouped):
            mi_subset = group[available_mi_cols].values
            
            mean_profile = np.mean(mi_subset, axis=0)
            std_profile = np.std(mi_subset, axis=0)
            sem_profile = std_profile / np.sqrt(len(group))
            
            color = self.palette[idx % len(self.palette)]
            
            # Plot mean line
            plt.plot(lags, mean_profile, label=str(disease), color=color, linewidth=2)
            # Plot standard error shaded region
            plt.fill_between(
                lags, 
                mean_profile - 1.96 * sem_profile, 
                mean_profile + 1.96 * sem_profile, 
                color=color, 
                alpha=0.15
            )
            
        plt.title("Average Mutual Information (AMI) Profile by Disease", pad=15)
        plt.xlabel("Lag (\u03c4)")
        plt.ylabel("Mutual Information (bits)")
        plt.legend(title="Disease Class")
        plt.xlim(1, len(available_mi_cols))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("MI profile plots saved successfully.")

    def plot_projections(self, df_features: pd.DataFrame, output_path: str, max_points: int = 2000) -> None:
        """
        Performs standard scaling and dimensionality reduction:
        - PCA (2 components)
        - UMAP or t-SNE (2 components)
        Generates 2D scatter plots colored by disease class.
        """
        self._ensure_dir(output_path)
        logger.info(f"Plotting 2D projections to {output_path}...")
        
        # Extract features (exclude identifier and text columns)
        exclude_cols = ["sample_id", "sequence", "disease", "species", "chromosome", 
                        "genomic_coordinates", "tissue", "sample_source", "eccdna_type", 
                        "health_status", "library_type", "pubmed_id", "public_date", 
                        "experiment_prediction", "validation_strategies", "treatment", 
                        "oncogene", "function_characteristic", "remarks", "discard_reason"]
        
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No numeric features found to project. Skipping projection plots.")
            return

        # Subsample to speed up non-linear projections
        if len(df_features) > max_points:
            logger.info(f"Subsampling to {max_points} points for projection visualization...")
            df_subset = df_features.sample(n=max_points, random_state=42).copy()
        else:
            df_subset = df_features.copy()

        X = df_subset[feature_cols].values
        labels = df_subset["disease"].values

        # 1. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. PCA Projection
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_ * 100

        # 3. Non-linear Projection (UMAP if installed, else t-SNE)
        proj_type = "UMAP" if HAS_UMAP else "t-SNE"
        logger.info(f"Computing {proj_type} projection...")
        
        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_2d = reducer.fit_transform(X_scaled)
        else:
            n_samples = X_scaled.shape[0]
            perplexity = min(30, max(1, n_samples - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
            X_2d = tsne.fit_transform(X_scaled)

        # 4. Generate Plot
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        # PCA plot
        sns.scatterplot(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            hue=labels, 
            ax=axes[0], 
            palette=self.palette,
            alpha=0.6,
            s=40
        )
        axes[0].set_title("PCA Projection (Linear)", pad=15)
        axes[0].set_xlabel(f"PC1 ({var_explained[0]:.2f}% var)")
        axes[0].set_ylabel(f"PC2 ({var_explained[1]:.2f}% var)")
        axes[0].legend(title="Disease")

        # UMAP / t-SNE plot
        sns.scatterplot(
            x=X_2d[:, 0], 
            y=X_2d[:, 1], 
            hue=labels, 
            ax=axes[1], 
            palette=self.palette,
            alpha=0.6,
            s=40
        )
        axes[1].set_title(f"{proj_type} Projection (Non-linear)", pad=15)
        axes[1].set_xlabel(f"{proj_type} Dimension 1")
        axes[1].set_ylabel(f"{proj_type} Dimension 2")
        axes[1].legend(title="Disease")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Projection plots saved successfully.")

    def detect_outliers(self, df_features: pd.DataFrame, contamination: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Uses an Isolation Forest to detect anomalous/outlier sequence profiles (top 1%).
        Returns:
            - normal_df: DataFrame with standard samples.
            - outlier_df: DataFrame containing flagged outliers.
        """
        logger.info(f"Starting outlier detection (Isolation Forest, contamination={contamination})...")
        
        exclude_cols = ["sample_id", "sequence", "disease", "species", "chromosome", 
                        "genomic_coordinates", "tissue", "sample_source", "eccdna_type", 
                        "health_status", "library_type", "pubmed_id", "public_date", 
                        "experiment_prediction", "validation_strategies", "treatment", 
                        "oncogene", "function_characteristic", "remarks", "discard_reason"]
        
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No numeric features found to run outlier detection.")
            return df_features, pd.DataFrame()

        # Handle NaNs if any exist
        X = df_features[feature_cols].fillna(0.0).values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        # Predict: 1 = normal, -1 = outlier
        preds = clf.fit_predict(X_scaled)
        
        df_copy = df_features.copy()
        df_copy["anomaly_score"] = clf.decision_function(X_scaled)
        df_copy["is_outlier"] = preds == -1

        normal_df = df_copy[df_copy["is_outlier"] == False].drop(columns=["is_outlier"]).reset_index(drop=True)
        outlier_df = df_copy[df_copy["is_outlier"] == True].reset_index(drop=True)

        logger.info(f"Outlier detection completed. Flagged {len(outlier_df)} outliers out of {len(df_features)} records.")
        return normal_df, outlier_df
