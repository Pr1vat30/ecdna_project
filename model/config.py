import torch

class ModelConfig:
    # Paths
    dataset_path = "datasets/eccDNA_sequences_clean.csv"
    
    # Feature names
    mi_cols = [f"MI_tau_{i}" for i in range(1, 101)]
    resolved_cols = [f"MI_Resolved_{base}_{tau}" for base in ['A', 'C', 'G', 'T'] for tau in range(1, 6)]
    ami_cols = ["AMI_1_20", "AMI_21_50", "AMI_51_100"]
    compositional_cols = ["sequence_length", "GC_pct"]
    kmer_3_cols = [f"kmer_3_{n1}{n2}{n3}" for n1 in 'ACGT' for n2 in 'ACGT' for n3 in 'ACGT']
    
    feature_cols = compositional_cols + ami_cols + resolved_cols + mi_cols + kmer_3_cols
    label_col = "disease"
    
    # Dataset splitting
    test_size = 0.15
    val_size = 0.15
    seed = 42
    
    # Sampler settings
    P = 7  # Number of classes per batch
    K = 64 # Samples per class
    
    # Network dimensions
    input_dim = len(feature_cols)
    hidden_dims = [256, 128]
    embed_dim = 64
    classifier_hidden = 128
    dropout = 0.3
    
    # Optimizer settings
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 50
    
    # Loss settings
    temperature = 0.07  # SupCon temperature
    lambda_max = 1.0    # Initial weight for metric loss
    lambda_min = 0.1    # Minimum weight for metric loss
    lambda_decay = 10.0 # Time constant for decay (in epochs)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
