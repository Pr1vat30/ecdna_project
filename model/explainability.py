import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

class eccDNAExplainer:
    """
    Computes SHAP feature attributions on top of the model's classification head, 
    generating summary charts linking sequence features to model decisions.
    """
    def __init__(self, model, config, feature_names):
        self.model = model
        self.config = config
        self.feature_names = feature_names
        self.device = config.device
        
    def explain_predictions(self, train_features, test_features, num_bg=100, num_exp=50, save_dir="datasets/plots", save_name="shap_summary_plot.png"):
        self.model.eval()
        
        # Selection of background reference subset
        bg_idx = np.random.choice(len(train_features), size=min(num_bg, len(train_features)), replace=False)
        bg_data = torch.tensor(train_features[bg_idx], dtype=torch.float32).to(self.device)
        
        # Selection of explanation query targets
        exp_idx = np.random.choice(len(test_features), size=min(num_exp, len(test_features)), replace=False)
        exp_data = torch.tensor(test_features[exp_idx], dtype=torch.float32).to(self.device)
        
        # Wrapping model to return only the classification head logits
        class ModelLogitWrapper(torch.nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner_model = inner_model
            def forward(self, x):
                _, logits = self.inner_model(x)
                return logits
                
        logit_model = ModelLogitWrapper(self.model).to(self.device)
        
        # Initialize DeepExplainer
        explainer = shap.DeepExplainer(logit_model, bg_data)
        
        # Calculate attributions (disable additivity checks for BatchNorm/L2Norm numerical variations)
        shap_values = explainer.shap_values(exp_data, check_additivity=False)
        
        # Convert explanation tensors to CPU/Numpy for matplotlib
        exp_data_np = exp_data.cpu().numpy()
        
        # Generate summary plot
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, exp_data_np, feature_names=self.feature_names, show=False
        )
        plt.title("SHAP Feature Importance (Softmax Classification Head)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values, exp_data_np
