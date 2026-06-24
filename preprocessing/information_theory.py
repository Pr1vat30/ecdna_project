import numpy as np
from typing import Dict, Any, Tuple, Optional
from preprocessing.config import PreprocessingConfig

class InformationTheoryCalculator:
    """
    Step 4: Information-Theoretic Feature Extraction
    Computes optimized, Laplace-smoothed Mutual Information (MI), Average Mutual Information (AMI),
    and Resolved Mutual Information (RMI) profiles.
    """
    def __init__(self, config: PreprocessingConfig, alpha: float = 0.1):
        self.config = config
        self.alpha = alpha  # Laplace smoothing pseudocount
        self.bases = ['A', 'C', 'G', 'T']
        self.base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def sequence_to_int_array(self, seq: str) -> np.ndarray:
        """
        Maps sequence characters to integer indices (A=0, C=1, G=2, T=3).
        Ambiguous bases or unknown characters are mapped to -1.
        """
        return np.array([self.base_to_idx.get(b, -1) for b in seq], dtype=np.int32)

    def calculate_mi_profiles(self, seq: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Calculates the Mutual Information (MI) and Resolved Mutual Information (RMI) profiles
        for lags 1 to T_adapt, where T_adapt = min(max_lag, L // 10).
        
        Returns:
            - mi_profile: 1D array of shape (T_adapt,) with total MI at each lag (in bits)
            - rmi_profile: 2D array of shape (4, T_adapt) with RMI values for [A, C, G, T] (in bits)
            - T_adapt: The actual lag limit used for this sequence.
        """
        n = len(seq)
        int_seq = self.sequence_to_int_array(seq)
        
        # Adaptive lag selection
        T_adapt = min(self.config.max_lag, n // 10)
        if T_adapt < 1:
            return np.zeros(0), np.zeros((4, 0)), 0

        mi_profile = np.zeros(T_adapt)
        rmi_profile = np.zeros((4, T_adapt))

        for tau in range(1, T_adapt + 1):
            # Mask out ambiguous bases (value is -1)
            mask = (int_seq[:-tau] >= 0) & (int_seq[tau:] >= 0)
            X = int_seq[:-tau][mask]
            Y = int_seq[tau:][mask]
            
            total_pairs = len(X)
            if total_pairs == 0:
                # If no valid pairs, keep profile at 0
                continue

            # Compute joint frequency matrix (4x4)
            # X * 4 + Y maps pairs to flat indices in 0..15
            joint_idx = X * 4 + Y
            counts = np.bincount(joint_idx, minlength=16).reshape((4, 4))
            
            # Laplace smoothing (pseudocounts)
            p_xy = (counts + self.alpha) / (total_pairs + 16 * self.alpha)
            
            # Compute marginals directly from joint probabilities to ensure sum to 1.0
            p_x = np.sum(p_xy, axis=1) # Row sums (X marginals)
            p_y = np.sum(p_xy, axis=0) # Column sums (Y marginals)
            
            # Outer product of marginals: p_x(a) * p_y(b)
            p_x_p_y = np.outer(p_x, p_y)
            
            # Compute Mutual Information term in bits (base 2)
            # Safe because p_xy > 0 and p_x_p_y > 0 due to Laplace smoothing
            mi_terms = p_xy * np.log2(p_xy / p_x_p_y)
            
            # Total MI for this lag
            mi_profile[tau - 1] = np.sum(mi_terms)
            
            # Resolved MI (decomposed by starting base x, axis=1 is the sum over y)
            rmi_profile[:, tau - 1] = np.sum(mi_terms, axis=1)

        return mi_profile, rmi_profile, T_adapt

    def compute_ami_features(self, mi_profile: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes Average Mutual Information (AMI) and Area Under the Curve (AUC)
        for pre-defined lag bands.
        
        Returns:
            - ami_vals: Dictionary of average MI in each band.
            - auc_vals: Dictionary of AUC in each band.
        """
        ami_vals = {}
        auc_vals = {}
        T = len(mi_profile)

        for start, end in self.config.ami_bands:
            band_name = f"{start}_{end}"
            
            # Convert to 0-based indices
            idx_start = start - 1
            idx_end = min(end, T)
            
            if idx_start >= T or idx_start >= idx_end:
                # If band lies beyond the adaptive lag limit, fill with 0.0
                ami_vals[f"AMI_{band_name}"] = 0.0
                auc_vals[f"AUC_{band_name}"] = 0.0
                continue
                
            segment = mi_profile[idx_start:idx_end]
            
            # 1. Average MI
            ami_vals[f"AMI_{band_name}"] = float(np.mean(segment))
            
            # 2. AUC using Trapezoidal rule
            if len(segment) > 1:
                auc_val = float(np.sum((segment[:-1] + segment[1:]) / 2.0))
            elif len(segment) == 1:
                auc_val = float(segment[0])
            else:
                auc_val = 0.0
            auc_vals[f"AUC_{band_name}"] = auc_val

        return ami_vals, auc_vals

    def compute_derived_mi_features(self, mi_profile: np.ndarray) -> Dict[str, float]:
        """
        Extracts derived shape and trend statistics from the MI profile.
        """
        if len(mi_profile) == 0:
            return {
                "mi_max": 0.0, "mi_argmax_lag": 0.0, "mi_mean": 0.0, "mi_median": 0.0,
                "mi_var": 0.0, "mi_std": 0.0, "mi_cv": 0.0, "mi_energy": 0.0,
                "mi_entropy": 0.0, "mi_decay_slope": 0.0
            }

        mi_max = float(np.max(mi_profile))
        # argmax lag is converted to 1-based lag
        mi_argmax_lag = float(np.argmax(mi_profile) + 1)
        mi_mean = float(np.mean(mi_profile))
        mi_median = float(np.median(mi_profile))
        mi_var = float(np.var(mi_profile))
        mi_std = float(np.std(mi_profile))
        mi_cv = mi_std / mi_mean if mi_mean > 0 else 0.0
        
        # 1. Profile Energy: sum of squares
        mi_energy = float(np.sum(mi_profile ** 2))
        
        # 2. Profile Entropy: treat normalized MI as a probability distribution
        total_mi = np.sum(mi_profile)
        if total_mi > 0:
            p_mi = mi_profile / total_mi
            # Add small epsilon to avoid log2(0) if any MI is exactly 0
            mi_entropy = float(-np.sum(p_mi * np.log2(p_mi + 1e-12)))
        else:
            mi_entropy = 0.0
            
        # 3. Decay Slope: simple linear regression slope of MI vs Lag
        lags = np.arange(1, len(mi_profile) + 1)
        if len(mi_profile) > 1:
            cov = np.cov(lags, mi_profile)[0, 1]
            var_lags = np.var(lags, ddof=1)
            mi_decay_slope = float(cov / var_lags) if var_lags > 0 else 0.0
        else:
            mi_decay_slope = 0.0

        return {
            "mi_max": mi_max,
            "mi_argmax_lag": mi_argmax_lag,
            "mi_mean": mi_mean,
            "mi_median": mi_median,
            "mi_var": mi_var,
            "mi_std": mi_std,
            "mi_cv": mi_cv,
            "mi_energy": mi_energy,
            "mi_entropy": mi_entropy,
            "mi_decay_slope": mi_decay_slope
        }
