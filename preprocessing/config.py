from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreprocessingConfig:
    # File Paths
    raw_metadata_path: str = "./datasets/eccDNA.csv"
    reference_genome_path: str = "./datasets/hg19.fa"
    output_dir: str = "./datasets"
    
    # Sequence filters
    max_seq_len: int = 100000
    min_seq_len: int = 100
    
    # Feature extraction settings
    max_lag: int = 100
    resolved_lag: int = 5
    kmer_k_values: List[int] = field(default_factory=lambda: [3, 4])
    ami_bands: List[Tuple[int, int]] = field(default_factory=lambda: [(1, 20), (21, 50), (51, 100)])
    
    # Disease filtering
    min_disease_samples: int = 1000
    max_per_class: int = 10000
