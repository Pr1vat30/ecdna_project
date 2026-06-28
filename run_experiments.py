import os
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from model.config import ModelConfig
from model.dataset import DatasetBuilder, PKBatchSampler, eccDNADataset
from model.networks import eccDNAModel
from model.trainer import SiameseTrainer
from model.evaluator import eccDNAEvaluator
from model.explainability import eccDNAExplainer

def slice_dataset(dataset, feature_indices):
    """
    Slices an eccDNADataset by selecting only a subset of features.
    """
    return eccDNADataset(dataset.features[:, feature_indices], dataset.labels)

def get_top_shap_features(model, config, train_dataset, test_dataset, num_features=20, save_name="mlp_baseline_shap.png"):
    """
    Runs SHAP explainability on the model to select the top num_features.
    """
    print("\nComputing SHAP attributions to identify best features...")
    train_features_np = train_dataset.features.numpy()
    test_features_np = test_dataset.features.numpy()
    
    explainer = eccDNAExplainer(model, config, config.feature_cols)
    shap_values, _ = explainer.explain_predictions(
        train_features=train_features_np,
        test_features=test_features_np,
        num_bg=100,
        num_exp=50,
        save_dir="datasets/plots",
        save_name=save_name
    )
    
    if isinstance(shap_values, list):
        abs_shap = np.mean([np.abs(val) for val in shap_values], axis=0)
        mean_abs_shap = np.mean(abs_shap, axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        
    sorted_idx = np.argsort(-mean_abs_shap)
    top_features = [config.feature_cols[i] for i in sorted_idx[:num_features]]
    
    print(f"Top {num_features} SHAP Features selected:")
    for rank, feat in enumerate(top_features):
        print(f"  {rank+1}: {feat} (Importance: {mean_abs_shap[sorted_idx[rank]]:.6f})")
        
    return top_features

def run_linear_experiment(config, train_ds, val_ds, test_ds, feature_cols, name, class_names, evaluator):
    """
    Runs Logistic Regression baseline on a specific feature subset.
    """
    print(f"\nRunning Logistic Regression on: {name} ({len(feature_cols)} features)...")
    
    # Map feature names to indices
    feature_to_idx = {feat: idx for idx, feat in enumerate(config.feature_cols)}
    feature_indices = [feature_to_idx[feat] for feat in feature_cols]
    
    X_train = train_ds.features[:, feature_indices].numpy()
    y_train = train_ds.labels.numpy()
    X_test = test_ds.features[:, feature_indices].numpy()
    y_test = test_ds.labels.numpy()
    
    lr = LogisticRegression(max_iter=1000, random_state=config.seed)
    lr.fit(X_train, y_train)
    
    preds = lr.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    # Detailed report print
    print(f"--- REPORT DETTAGLIATO PER {name.upper()} ---")
    try:
        report_text_det = classification_report(y_test, preds, target_names=class_names, zero_division=0.0)
    except Exception:
        report_text_det = classification_report(y_test, preds, zero_division=0.0)
    print(report_text_det)
    
    # Calculate retrieval metrics using decision boundaries
    logits = lr.decision_function(X_test)
    norms = np.linalg.norm(logits, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    logits_norm = logits / norms
    
    retrieval_metrics = evaluator.evaluate_retrieval(logits_norm, y_test)
    
    return {
        "Accuracy": accuracy,
        "Recall@1": retrieval_metrics["Recall@1"],
        "Recall@5": retrieval_metrics["Recall@5"],
        "mAP": retrieval_metrics["mAP"],
        "DetailedReport": report_text_det
    }

def run_nn_experiment(config, train_ds, val_ds, test_ds, feature_cols, name, model_type, checkpoint_path, class_names, label_encoder):
    """
    Runs training and evaluation for Deep MLP or CNN model on a specific feature subset.
    """
    print(f"\n--- Running Experiment: {name} ({model_type.upper()} with {len(feature_cols)} features) ---")
    
    # Map feature names to indices
    feature_to_idx = {feat: idx for idx, feat in enumerate(config.feature_cols)}
    feature_indices = [feature_to_idx[feat] for feat in feature_cols]
    
    # Slice datasets
    train_sliced = slice_dataset(train_ds, feature_indices)
    val_sliced = slice_dataset(val_ds, feature_indices)
    test_sliced = slice_dataset(test_ds, feature_indices)
    
    num_classes = len(class_names)
    
    # Setup temporary config
    class TempConfig(ModelConfig):
        pass
    temp_config = TempConfig()
    temp_config.feature_cols = feature_cols
    temp_config.input_dim = len(feature_cols)
    temp_config.model_type = model_type
    
    # Instantiate Model
    model = eccDNAModel(temp_config, num_classes=num_classes)
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    else:
        print(f"No checkpoint found at {checkpoint_path}. Training from scratch...")
        train_sampler = PKBatchSampler(train_sliced.labels.numpy(), P=config.P, K=config.K)
        train_loader = torch.utils.data.DataLoader(train_sliced, batch_sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_sliced, batch_size=256, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        trainer = SiameseTrainer(model, train_loader, val_loader, optimizer, config, checkpoint_path=checkpoint_path)
        trainer.fit()
        
        # Load best weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    
    model = model.to(config.device)
    model.eval()
    
    # Evaluate Retrieval
    test_loader = torch.utils.data.DataLoader(test_sliced, batch_size=256, shuffle=False)
    evaluator = eccDNAEvaluator(model, temp_config, label_encoder)
    test_embeddings, test_labels = evaluator.extract_embeddings(test_loader)
    retrieval_metrics = evaluator.evaluate_retrieval(test_embeddings, test_labels)
    
    if name in ["MLP_baseline", "CNN_baseline"]:
        plot_path = f"datasets/plots/{name.lower()}_embeddings.png"
        print(f"Generating PCA and t-SNE projections to {plot_path}...")
        evaluator.plot_embeddings(test_embeddings, test_labels, plot_path)
    
    # Evaluate Classification
    all_preds = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(config.device)
            _, logits = model(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    accuracy = accuracy_score(test_labels, all_preds)
    
    # Detailed report print
    print(f"--- REPORT DETTAGLIATO PER {name.upper()} ---")
    try:
        report_text_det = classification_report(test_labels, all_preds, target_names=class_names, zero_division=0.0)
    except Exception:
        report_text_det = classification_report(test_labels, all_preds, zero_division=0.0)
    print(report_text_det)
    # Run SHAP explanation if requested or for specific experiments
    if name in ["MLP_info_theoretic", "CNN_baseline", "CNN_info_theoretic"]:
        print(f"\nComputing SHAP attributions for {name}...")
        explainer = eccDNAExplainer(model, temp_config, feature_cols)
        X_train_sliced = train_sliced.features.numpy()
        X_test_sliced = test_sliced.features.numpy()
        explainer.explain_predictions(
            train_features=X_train_sliced,
            test_features=X_test_sliced,
            num_bg=100,
            num_exp=50,
            save_dir="datasets/plots",
            save_name=f"{name.lower()}_shap.png"
        )
        
    return {
        "Accuracy": accuracy,
        "Recall@1": retrieval_metrics["Recall@1"],
        "Recall@5": retrieval_metrics["Recall@5"],
        "mAP": retrieval_metrics["mAP"],
        "DetailedReport": report_text_det
    }

def main():
    print("=== STARTING CONSOLIDATED eccDNA EXPERIMENTS ORCHESTRATOR ===")
    config = ModelConfig()
    
    # 1. Ingest and scale data ONCE
    print("Loading and splitting dataset...")
    builder = DatasetBuilder(config)
    train_dataset, val_dataset, test_dataset = builder.build_datasets()
    
    class_names = list(builder.label_encoder.classes_)
    num_classes = len(class_names)
    print(f"Dataset split completed successfully.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Number of classes (diseases): {num_classes}")
    for idx, name in enumerate(class_names):
        print(f"  Class {idx}: {name}")
        
    evaluator = eccDNAEvaluator(None, config, builder.label_encoder)
    
    # 2. Train baseline MLP model if not already trained, and get Top 20 SHAP features
    mlp_baseline_path = "models/best_mlp_baseline.pt"
    baseline_model = eccDNAModel(config, num_classes=num_classes)
    
    if os.path.exists(mlp_baseline_path):
        print(f"\nLoading existing MLP Baseline model from {mlp_baseline_path} for SHAP feature selection...")
        baseline_model.load_state_dict(torch.load(mlp_baseline_path, map_location=config.device))
    else:
        print(f"\nTraining MLP Baseline model {len(config.feature_cols)} features to compute SHAP attributions...")
        train_sampler = PKBatchSampler(train_dataset.labels.numpy(), P=config.P, K=config.K)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        trainer = SiameseTrainer(baseline_model, train_loader, val_loader, optimizer, config, checkpoint_path=mlp_baseline_path)
        trainer.fit()
        baseline_model.load_state_dict(torch.load(mlp_baseline_path, map_location=config.device))
        
    baseline_model = baseline_model.to(config.device)
    top_20_shap = get_top_shap_features(baseline_model, config, train_dataset, test_dataset, num_features=20, save_name="mlp_baseline_shap.png")
    
    # 3. Define the three feature subsets
    feature_subsets = {
        "baseline": (config.feature_cols, f"Baseline (All {len(config.feature_cols)} features)"),
        "shap_selected": (top_20_shap, "SHAP Selected (Top 20 features)"),
        "info_theoretic": (config.compositional_cols[1:] + config.ami_cols + config.resolved_cols + config.mi_cols, "Info-Theoretic (GC+MI+AMI+Resolved)")
    }
    
    results = {}
    
    # 4. Run Logistic Regression Experiments
    for key, (features, label_name) in feature_subsets.items():
        exp_name = f"Linear_{key}"
        results[exp_name] = run_linear_experiment(
            config, train_dataset, val_dataset, test_dataset, features, exp_name, class_names, evaluator
        )
        
    # 5. Run MLP Siamese Model Experiments
    # The baseline MLP model on all features is already trained/loaded
    print(f"\nRunning MLP Siamese Model Baseline evaluation (MLP with {len(config.feature_cols)} features)")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    evaluator.model = baseline_model
    baseline_embeddings, baseline_labels = evaluator.extract_embeddings(test_loader)
    baseline_retrieval = evaluator.evaluate_retrieval(baseline_embeddings, baseline_labels)
    
    # Generate MLP Baseline Embedding Plot
    plot_path = "datasets/plots/mlp_baseline_embeddings.png"
    print(f"Generating PCA and t-SNE projections for MLP Baseline to {plot_path}...")
    evaluator.plot_embeddings(baseline_embeddings, baseline_labels, plot_path)
    
    all_preds = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(config.device)
            _, logits = baseline_model(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    baseline_accuracy = accuracy_score(baseline_labels, all_preds)
    
    print(f"--- REPORT DETTAGLIATO PER MLP_BASELINE ---")
    try:
        baseline_report_det = classification_report(baseline_labels, all_preds, target_names=class_names, zero_division=0.0)
    except Exception:
        baseline_report_det = classification_report(baseline_labels, all_preds, zero_division=0.0)
    print(baseline_report_det)
    
    results["MLP_baseline"] = {
        "Accuracy": baseline_accuracy,
        "Recall@1": baseline_retrieval["Recall@1"],
        "Recall@5": baseline_retrieval["Recall@5"],
        "mAP": baseline_retrieval["mAP"],
        "DetailedReport": baseline_report_det
    }
    
    # Run the remaining MLP feature subsets
    for key in ["shap_selected", "info_theoretic"]:
        features, _ = feature_subsets[key]
        exp_name = f"MLP_{key}"
        checkpoint_path = f"models/best_mlp_{key}.pt"
        results[exp_name] = run_nn_experiment(
            config, train_dataset, val_dataset, test_dataset, features, exp_name,
            model_type="mlp", checkpoint_path=checkpoint_path,
            class_names=class_names, label_encoder=builder.label_encoder
        )
        
    # 6. Run 1D-CNN Siamese Model Experiments
    for key, (features, _) in feature_subsets.items():
        exp_name = f"CNN_{key}"
        checkpoint_path = f"models/best_cnn_{key}.pt"
        results[exp_name] = run_nn_experiment(
            config, train_dataset, val_dataset, test_dataset, features, exp_name,
            model_type="cnn", checkpoint_path=checkpoint_path,
            class_names=class_names, label_encoder=builder.label_encoder
        )
        
    # 7. Print and Save Consolidated Report
    report_lines = []
    report_lines.append("==========================================================================================")
    report_lines.append("Model & Feature Subset                         Accuracy    Recall@1    Recall@5      mAP")
    report_lines.append("------------------------------------------------------------------------------------------")
    
    # Print function helper
    def add_result_line(exp_key, display_name):
        r = results[exp_key]
        report_lines.append(f"{display_name:<46} {r['Accuracy']:.4f}      {r['Recall@1']:.4f}      {r['Recall@5']:.4f}      {r['mAP']:.4f}")
        
    # Linear Baselines
    add_result_line("Linear_baseline", f"Logistic Regression - Baseline ({len(config.feature_cols)} feats)")
    add_result_line("Linear_shap_selected", "Logistic Regression - SHAP Top 20")
    add_result_line("Linear_info_theoretic", "Logistic Regression - Info-Theoretic (124 feats)")
    report_lines.append("------------------------------------------------------------------------------------------")
    
    # MLP Models
    add_result_line("MLP_baseline", f"MLP Siamese Model - Baseline ({len(config.feature_cols)} feats)")
    add_result_line("MLP_shap_selected", "MLP Siamese Model - SHAP Top 20")
    add_result_line("MLP_info_theoretic", "MLP Siamese Model - Info-Theoretic (124 feats)")
    report_lines.append("------------------------------------------------------------------------------------------")
    
    # CNN Models
    add_result_line("CNN_baseline", f"1D-CNN Siamese Model - Baseline ({len(config.feature_cols)} feats)")
    add_result_line("CNN_shap_selected", "1D-CNN Siamese Model - SHAP Top 20")
    add_result_line("CNN_info_theoretic", "1D-CNN Siamese Model - Info-Theoretic (124 feats)")
    
    report_lines.append("==========================================================================================")
    
    comparison_text = "\n".join(report_lines)
    print("\n--- ALL EXPERIMENTS COMPARISON SUMMARY ---")
    print(comparison_text)
    
    # Save unified markdown report
    os.makedirs("datasets/reports", exist_ok=True)
    report_path = "datasets/reports/experiments_summary_report.md"
    with open(report_path, "w") as f:
        f.write("# eccDNA Orchestrated Experiment Runner Summary Report\n\n")
        
        f.write("## Summary Table\n\n```\n")
        f.write(comparison_text)
        f.write("\n```\n\n")
        
        f.write("## Selected Top 20 SHAP Features\n\n")
        for rank, feat in enumerate(top_20_shap):
            f.write(f"{rank+1}. **{feat}**\n")
        f.write("\n")
        
        f.write("## Detailed Classification Reports\n\n")
        for exp_key, r in results.items():
            f.write(f"### {exp_key}\n")
            f.write("```\n")
            f.write(r['DetailedReport'])
            f.write("```\n\n")
            
    print(f"\nConsolidated experiments summary report saved to {report_path}")
    print("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
