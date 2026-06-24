import math
import logging
import itertools
from typing import Dict, Any, List, Tuple
import numpy as np
from preprocessing.config import PreprocessingConfig
from preprocessing.information_theory import InformationTheoryCalculator

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Step 4: Sequence Feature Extraction Pipeline
    Orchestrates the computation of basic compositional, information-theoretic,
    and optional alignment-free k-mer features for a DNA sequence.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.info_calc = InformationTheoryCalculator(config)
        self.bases = ['A', 'C', 'G', 'T']
        
        # Pre-build string names for dinucleotides and k-mers to map flat indices to clear feature names
        self.dinuc_names = [f"freq_{b1}{b2}" for b1 in self.bases for b2 in self.bases]
        self.kmer3_names = [f"kmer_3_{''.join(p)}" for p in itertools.product(self.bases, repeat=3)]
        self.kmer4_names = [f"kmer_4_{''.join(p)}" for p in itertools.product(self.bases, repeat=4)]

    def extract_compositional_features(self, seq: str) -> Dict[str, float]:
        """
        Computes GC/AT content, skews, single nucleotide frequencies,
        vectorized dinucleotide frequencies, and Shannon entropy.
        """
        features = {}
        L = len(seq)
        if L == 0:
            return {}

        int_seq = self.info_calc.sequence_to_int_array(seq)
        valid_mask = int_seq >= 0
        valid_len = np.sum(valid_mask)
        
        if valid_len == 0:
            # If sequence contains only invalid bases
            features["GC_pct"] = 0.0
            features["AT_pct"] = 0.0
            features["GC_skew"] = 0.0
            features["AT_skew"] = 0.0
            features["shannon_entropy"] = 0.0
            features["freq_A"] = 0.0
            features["freq_C"] = 0.0
            features["freq_G"] = 0.0
            features["freq_T"] = 0.0
            for name in self.dinuc_names:
                features[name] = 0.0
            return features

        # 1. Mononucleotide counts
        counts_1 = np.bincount(int_seq[valid_mask], minlength=4)
        freq_1 = counts_1 / valid_len
        
        c_A, c_C, c_G, c_T = counts_1[0], counts_1[1], counts_1[2], counts_1[3]
        
        # GC/AT content
        features["GC_pct"] = float((c_G + c_C) / valid_len * 100)
        features["AT_pct"] = float((c_A + c_T) / valid_len * 100)
        
        # Skews
        features["GC_skew"] = float((c_G - c_C) / (c_G + c_C)) if (c_G + c_C) > 0 else 0.0
        features["AT_skew"] = float((c_A - c_T) / (c_A + c_T)) if (c_A + c_T) > 0 else 0.0
        
        # Frequencies
        features["freq_A"] = float(freq_1[0])
        features["freq_C"] = float(freq_1[1])
        features["freq_G"] = float(freq_1[2])
        features["freq_T"] = float(freq_1[3])
        
        # Shannon Entropy (base 2)
        entropy = -np.sum([p * np.log2(p) for p in freq_1 if p > 0])
        features["shannon_entropy"] = float(entropy)

        # 2. Vectorized Dinucleotide frequencies
        mask_2 = (int_seq[:-1] >= 0) & (int_seq[1:] >= 0)
        total_2 = np.sum(mask_2)
        if total_2 > 0:
            idx_2 = int_seq[:-1][mask_2] * 4 + int_seq[1:][mask_2]
            counts_2 = np.bincount(idx_2, minlength=16)
            freq_2 = counts_2 / total_2
        else:
            freq_2 = np.zeros(16)
            
        for i, name in enumerate(self.dinuc_names):
            features[name] = float(freq_2[i])

        return features

    def extract_kmer_features(self, seq: str) -> Dict[str, float]:
        """
        Computes frequencies of 3-mers (64 features) and 4-mers (256 features)
        in a fully vectorized manner.
        """
        features = {}
        int_seq = self.info_calc.sequence_to_int_array(seq)
        
        # 3-mers
        mask_3 = (int_seq[:-2] >= 0) & (int_seq[1:-1] >= 0) & (int_seq[2:] >= 0)
        total_3 = np.sum(mask_3)
        if total_3 > 0:
            idx_3 = int_seq[:-2][mask_3] * 16 + int_seq[1:-1][mask_3] * 4 + int_seq[2:][mask_3]
            freq_3 = np.bincount(idx_3, minlength=64) / total_3
        else:
            freq_3 = np.zeros(64)
            
        for i, name in enumerate(self.kmer3_names):
            features[name] = float(freq_3[i])

        # 4-mers
        mask_4 = (int_seq[:-3] >= 0) & (int_seq[1:-2] >= 0) & (int_seq[2:-1] >= 0) & (int_seq[3:] >= 0)
        total_4 = np.sum(mask_4)
        if total_4 > 0:
            idx_4 = int_seq[:-3][mask_4] * 64 + int_seq[1:-2][mask_4] * 16 + int_seq[2:-1][mask_4] * 4 + int_seq[3:][mask_4]
            freq_4 = np.bincount(idx_4, minlength=256) / total_4
        else:
            freq_4 = np.zeros(256)
            
        for i, name in enumerate(self.kmer4_names):
            features[name] = float(freq_4[i])

        return features

    def extract_all_features(self, seq: str, sample_id: str, genomic_len: int, include_kmers: bool = True) -> Dict[str, Any]:
        """
        Extracts the complete feature dictionary for a single sequence.
        """
        record = {
            "sample_id": sample_id,
            "sequence_length": len(seq),
            "genomic_length": genomic_len
        }

        # 1. Compositional features
        comp_features = self.extract_compositional_features(seq)
        record.update(comp_features)

        # 2. Information theory features (MI and RMI profiles)
        mi_profile, rmi_profile, T_actual = self.info_calc.calculate_mi_profiles(seq)

        # Pad MI profile to max_lag with 0s if sequence was shorter than max_lag * 10
        # This keeps the final feature vector size identical across sequences for downstream ML models.
        padded_mi = np.zeros(self.config.max_lag)
        if T_actual > 0:
            padded_mi[:T_actual] = mi_profile

        for tau in range(1, self.config.max_lag + 1):
            record[f"MI_tau_{tau}"] = float(padded_mi[tau - 1])

        # Pad and pack RMI features
        padded_rmi = np.zeros((4, self.config.resolved_lag))
        T_rmi_actual = min(self.config.resolved_lag, T_actual)
        if T_rmi_actual > 0:
            padded_rmi[:, :T_rmi_actual] = rmi_profile[:, :T_rmi_actual]

        for i, base in enumerate(self.bases):
            for tau in range(1, self.config.resolved_lag + 1):
                record[f"MI_Resolved_{base}_{tau}"] = float(padded_rmi[i, tau - 1])

        # 3. AMI summaries
        ami_vals, auc_vals = self.info_calc.compute_ami_features(padded_mi)
        record.update(ami_vals)
        record.update(auc_vals)

        # 4. Derived MI shape features
        derived_vals = self.info_calc.compute_derived_mi_features(padded_mi[:T_actual] if T_actual > 0 else np.zeros(0))
        record.update(derived_vals)

        # 5. Optional K-mer features
        if include_kmers:
            kmer_features = self.extract_kmer_features(seq)
            record.update(kmer_features)

        return record
