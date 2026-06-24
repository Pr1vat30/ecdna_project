#!/usr/bin/env python3
"""
Orchestration Script for eccDNA Preprocessing Pipeline.
Imports the modular preprocessing package and runs the full end-to-end
pipeline to reconstruct, clean, filter, and extract features from the dataset.
"""

from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.utils import setup_logging

def main():
    # 1. Initialize package logging
    setup_logging()
    
    # 2. Define the configuration parameters
    # Adjust paths and parameters as necessary here.
    config = PreprocessingConfig(
        raw_metadata_path="./datasets/eccDNA.csv",
        reference_genome_path="./datasets/hg19.fa",
        output_dir="./datasets",
        min_seq_len=100,          # Minimum sequence length to allow robust MI profiles
        max_seq_len=100000,       # Maximum sequence length to prevent memory issues
        max_lag=100,              # Maximum lag for Mutual Information
        resolved_lag=5,           # Maximum lag for Resolved Mutual Information
        min_disease_samples=1000  # Minimum sample size threshold for disease classes
    )
    
    # 3. Instantiate the pipeline orchestrator
    pipeline = PreprocessingPipeline(config)
    
    # 4. Execute the complete preprocessing pipeline
    # Set limit=None to run on the full dataset, or limit=5000 for a fast dry-run check.
    # Set include_kmers=True to extract 3-mer and 4-mer alignment-free features.
    clean_features_df, outlier_features_df = pipeline.run(limit=None, include_kmers=True)
    
    print("\n==================================================")
    print("Execution completed successfully!")
    print(f"Clean features matrix shape: {clean_features_df.shape}")
    print(f"Outlier records shape: {outlier_features_df.shape}")
    print("==================================================")

if __name__ == "__main__":
    main()
