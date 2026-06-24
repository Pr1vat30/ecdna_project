import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from model.config import ModelConfig
from model.dataset import DatasetBuilder
from model.networks import eccDNAModel
from model.evaluator import eccDNAEvaluator

def main():
    config = ModelConfig()
    
    # Build datasets
    builder = DatasetBuilder(config)
    train_dataset, val_dataset, test_dataset = builder.build_datasets()
    
    num_classes = len(builder.label_encoder.classes_)
    class_names = list(builder.label_encoder.classes_)
    
    # Loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    # Initialize Model and load best checkpoint
    model = eccDNAModel(config, num_classes=num_classes)
    best_model_path = "models/best_mlp_baseline.pt"
    if not os.path.exists(best_model_path):
        print(f"Error: Best model checkpoint not found at {best_model_path}")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    
    # Run classification inference
    all_preds = []
    all_targets = []
    all_embeddings = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(config.device)
            embeds, logits = model(features)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_embeddings.append(embeds.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Compute classification report
    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )
    
    # Compute retrieval metrics
    evaluator = eccDNAEvaluator(model, config, builder.label_encoder)
    retrieval_metrics = evaluator.evaluate_retrieval(all_embeddings, all_targets)
    
    # Create the report content
    report_lines = []
    report_lines.append("# eccDNA Model Performance & Configuration Report")
    report_lines.append("")
    report_lines.append("## 1. Model Configuration")
    report_lines.append("")
    report_lines.append("| Hyperparameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Input Feature Dimension | {config.input_dim} |")
    report_lines.append(f"| Hidden MLP Layers (Encoder) | {config.hidden_dims} |")
    report_lines.append(f"| Embedding Dimension | {config.embed_dim} |")
    report_lines.append(f"| Classifier Hidden Layer | {config.classifier_hidden} |")
    report_lines.append(f"| Dropout Rate | {config.dropout} |")
    report_lines.append(f"| Optimizer | Adam (lr={config.lr}, weight_decay={config.weight_decay}) |")
    report_lines.append(f"| Loss Balancing (Lambda Schedule) | Max={config.lambda_max}, Min={config.lambda_min}, Decay={config.lambda_decay} epochs |")
    report_lines.append(f"| P-K Batch Sampler | P={config.P} classes, K={config.K} samples (Batch Size={config.P * config.K}) |")
    report_lines.append(f"| Training Device | {config.device} |")
    report_lines.append(f"| Total Epochs | {config.epochs} |")
    report_lines.append("")
    report_lines.append("## 2. Embedding Space Retrieval Performance")
    report_lines.append("")
    report_lines.append("| Retrieval Metric | Score | Description |")
    report_lines.append("|---|---|---|")
    report_lines.append(f"| **Recall@1** | {retrieval_metrics['Recall@1']:.4f} | Percentage of queries matching the correct class in top-1 |")
    report_lines.append(f"| **Recall@5** | {retrieval_metrics['Recall@5']:.4f} | Percentage of queries matching the correct class in top-5 |")
    report_lines.append(f"| **Mean Average Precision (mAP)** | {retrieval_metrics['mAP']:.4f} | Mean ranking precision of positive matches across gallery |")
    report_lines.append("")
    report_lines.append("## 3. Classification Performance (Softmax Head)")
    report_lines.append("")
    report_lines.append("| Disease Class | Precision | Recall | F1-Score | Support |")
    report_lines.append("|---|---|---|---|---|")
    
    for c_name in class_names:
        metrics = report_dict[c_name]
        report_lines.append(f"| {c_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |")
        
    report_lines.append("|---|---|---|---|---|")
    
    # Add averages
    accuracy = report_dict['accuracy']
    macro_avg = report_dict['macro avg']
    weighted_avg = report_dict['weighted avg']
    
    report_lines.append(f"| **Accuracy** | | | {accuracy:.4f} | {len(all_targets)} |")
    report_lines.append(f"| **Macro Average** | {macro_avg['precision']:.4f} | {macro_avg['recall']:.4f} | {macro_avg['f1-score']:.4f} | {int(macro_avg['support'])} |")
    report_lines.append(f"| **Weighted Average** | {weighted_avg['precision']:.4f} | {weighted_avg['recall']:.4f} | {weighted_avg['f1-score']:.4f} | {int(weighted_avg['support'])} |")
    
    report_text = "\n".join(report_lines)
    
    # Print the report to the console
    print(report_text)
    
    # Save the report to reports directory
    os.makedirs("datasets/reports", exist_ok=True)
    report_path = "datasets/reports/model_performance_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport successfully saved to {report_path}")

if __name__ == "__main__":
    main()
