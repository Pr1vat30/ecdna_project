import logging
import math
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)

class DiseaseFilter:
    """
    Step 3: Disease Filtering
    Filters diseases with sufficient samples to ensure statistical power and valid downstream training.
    Generates structured reports summarizing frequencies and power.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def calculate_mde(self, sample_size: int, power: float = 0.80, alpha: float = 0.05) -> float:
        """
        Calculates the Minimum Detectable Effect (MDE) in terms of Cohen's d
        for a two-sample comparison between this disease class and another class of equal size.
        """
        if sample_size <= 0:
            return float('inf')
        
        # Approximate MDE formula: d = (z_{1-alpha/2} + z_{power}) * sqrt(2/n)
        # For alpha = 0.05 (two-tailed), z_{0.975} = 1.96
        # For power = 0.80, z_{0.80} = 0.84
        z_alpha = 1.96
        z_power = 0.84
        
        mde = (z_alpha + z_power) * math.sqrt(2.0 / sample_size)
        return mde

    def filter_and_report(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """
        Filters out diseases with samples less than min_disease_samples.
        Returns:
            - filtered_df: DataFrame containing only valid diseases.
            - report_dict: Dictionary containing summary statistics.
            - report_markdown: Markdown formatted string of the report.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to DiseaseFilter.")
            return df, {}, "# Disease Filtering Report\n\nEmpty input dataset."

        # Count frequencies
        counts = df["disease"].value_counts(dropna=False)
        
        min_threshold = self.config.min_disease_samples
        
        retained_diseases = []
        excluded_diseases = []
        
        retained_records_count = 0
        excluded_records_count = 0
        
        for disease, count in counts.items():
            mde = self.calculate_mde(count)
            disease_str = str(disease)
            
            info = {
                "disease": disease_str,
                "count": int(count),
                "mde_cohens_d": round(mde, 4)
            }
            
            if count >= min_threshold:
                retained_diseases.append(info)
                retained_records_count += count
            else:
                excluded_diseases.append(info)
                excluded_records_count += count
                
        # Filter the DataFrame
        valid_disease_names = [info["disease"] for info in retained_diseases]
        filtered_df = df[df["disease"].isin(valid_disease_names)].copy().reset_index(drop=True)
        
        # Build report dictionary
        report_dict = {
            "total_input_records": len(df),
            "total_retained_records": int(retained_records_count),
            "total_excluded_records": int(excluded_records_count),
            "num_retained_diseases": len(retained_diseases),
            "num_excluded_diseases": len(excluded_diseases),
            "retained_list": retained_diseases,
            "excluded_list": excluded_diseases
        }
        
        # Build report markdown
        md = []
        md.append("# Disease Filtering Report")
        md.append(f"**Total Input Records:** {len(df)}")
        md.append(f"**Total Retained Records:** {retained_records_count} ({retained_records_count/len(df)*100:.2f}%)")
        md.append(f"**Total Excluded Records:** {excluded_records_count} ({excluded_records_count/len(df)*100:.2f}%)")
        md.append(f"**Retained Disease Classes (Threshold \u2265 {min_threshold}):** {len(retained_diseases)}")
        md.append(f"**Excluded Disease Classes (Threshold < {min_threshold}):** {len(excluded_diseases)}\n")
        
        md.append("## Retained Diseases")
        if retained_diseases:
            md.append("| Disease Name | Sample Count | Min Detectable Effect (Cohen's d) |")
            md.append("|:---|:---:|:---:|")
            for item in retained_diseases:
                md.append(f"| {item['disease']} | {item['count']} | {item['mde_cohens_d']} |")
        else:
            md.append("*No diseases met the minimum sample size threshold.*")
            
        md.append("\n## Excluded Diseases (Sample)")
        # Show top 20 excluded diseases by count to keep report readable
        if excluded_diseases:
            md.append("| Disease Name | Sample Count | Min Detectable Effect (Cohen's d) |")
            md.append("|:---|:---:|:---:|")
            for item in excluded_diseases[:20]:
                md.append(f"| {item['disease']} | {item['count']} | {item['mde_cohens_d']} |")
            if len(excluded_diseases) > 20:
                md.append(f"\n*... and {len(excluded_diseases) - 20} more excluded diseases.*")
        else:
            md.append("*No diseases were excluded.*")
            
        report_markdown = "\n".join(md)
        
        logger.info(f"Disease filtering completed. Retained {len(retained_diseases)} diseases ({retained_records_count} records).")
        return filtered_df, report_dict, report_markdown


class DiseaseStatsAggregator:
    """
    Step 5: Disease-level Statistical Analysis
    Computes summary statistics (mean, std, SEM, analytical CI, and bootstrap CI)
    for key features across different disease classes.
    """
    def __init__(self, config: PreprocessingConfig, bootstrap_replicates: int = 1000):
        self.config = config
        self.bootstrap_replicates = bootstrap_replicates

    def compute_bootstrap_ci(self, values: np.ndarray) -> Tuple[float, float]:
        """Computes the 95% bootstrap confidence interval of the mean."""
        if len(values) == 0:
            return 0.0, 0.0
        B = self.bootstrap_replicates
        n = len(values)
        resamples = np.random.choice(values, size=(B, n), replace=True)
        boot_means = np.mean(resamples, axis=1)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        return ci_lower, ci_upper

    def aggregate_statistics(self, df_features: pd.DataFrame, features_to_aggregate: List[str] = None) -> pd.DataFrame:
        """
        Groups features by disease and calculates mean, standard deviation, SEM,
        analytical CI, and bootstrap CI.
        """
        import numpy as np
        if features_to_aggregate is None:
            # Default key features to aggregate
            features_to_aggregate = ["sequence_length", "GC_pct", "shannon_entropy"]
            # Add any AMI features that exist in the dataframe
            ami_cols = [c for c in df_features.columns if c.startswith("AMI_") or c.startswith("AUC_")]
            features_to_aggregate.extend(ami_cols[:6])
            
        records = []
        
        # Group by disease
        grouped = df_features.groupby("disease")
        
        for disease, group in grouped:
            for feat in features_to_aggregate:
                if feat not in group.columns:
                    continue
                    
                vals = pd.to_numeric(group[feat], errors='coerce').dropna().values
                n = len(vals)
                if n == 0:
                    continue
                    
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                sem = std / math.sqrt(n) if n > 0 else 0.0
                
                # Analytical 95% CI
                ci_lower = mean - 1.96 * sem
                ci_upper = mean + 1.96 * sem
                
                # Bootstrap 95% CI (computed for sequence_length or GC_pct to handle non-normality)
                if feat in ["sequence_length", "GC_pct"]:
                    boot_lower, boot_upper = self.compute_bootstrap_ci(vals)
                else:
                    boot_lower, boot_upper = ci_lower, ci_upper
                    
                records.append({
                    "disease": str(disease),
                    "feature": feat,
                    "sample_count": n,
                    "mean": mean,
                    "std_dev": std,
                    "sem": sem,
                    "ci_95_lower": ci_lower,
                    "ci_95_upper": ci_upper,
                    "bootstrap_ci_lower": boot_lower,
                    "bootstrap_ci_upper": boot_upper
                })
                
        return pd.DataFrame(records)

    def generate_publication_table(self, df_summary: pd.DataFrame) -> str:
        """Generates a publication-quality Markdown table of the aggregated statistics."""
        if df_summary.empty:
            return "No data to summarize."
            
        md = []
        md.append("# Disease-level Feature Statistics Summary\n")
        md.append("| Disease | Feature | Count | Mean | Std Dev | 95% CI (Analytical) | 95% CI (Bootstrap) |")
        md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
        
        for _, row in df_summary.iterrows():
            disease = row["disease"]
            feat = row["feature"]
            n = row["sample_count"]
            mean = f"{row['mean']:.4f}"
            std = f"{row['std_dev']:.4f}"
            ci = f"[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}]"
            
            # Display bootstrap CI if it was computed (different from analytical)
            if feat in ["sequence_length", "GC_pct"]:
                boot_ci = f"[{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}]"
            else:
                boot_ci = "N/A"
                
            md.append(f"| {disease} | {feat} | {n} | {mean} | {std} | {ci} | {boot_ci} |")
            
        return "\n".join(md)

