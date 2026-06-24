import os
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from preprocessing.config import PreprocessingConfig
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.sequence_cleaner import SequenceCleaner
from preprocessing.disease_statistics import DiseaseFilter, DiseaseStatsAggregator
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.visualization import EDAVisualizer
from preprocessing.utils import save_report

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Step 7: Software Package Orchestration
    Unified runner class that coordinates all preprocessing, cleaning,
    feature extraction, stats aggregation, plotting, and outlier detection stages.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Instantiate all sub-modules
        self.loader = DatasetLoader(config)
        self.cleaner = SequenceCleaner(config)
        self.disease_filter = DiseaseFilter(config)
        self.extractor = FeatureExtractor(config)
        self.stats_aggregator = DiseaseStatsAggregator(config)
        self.visualizer = EDAVisualizer(config)

    def run(self, limit: Optional[int] = None, include_kmers: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the full end-to-end preprocessing pipeline.
        Returns:
            - clean_features_df: DataFrame containing the final cleaned features dataset.
            - outlier_features_df: DataFrame containing the isolated outlier records.
        """
        logger.info("==================================================")
        logger.info("STARTING END-TO-END PREPROCESSING PIPELINE")
        logger.info("==================================================")

        # --------------------------------------------------
        # Step 1: Dataset Construction
        # --------------------------------------------------
        logger.info("--- STAGE 1: Dataset Construction ---")
        raw_df = self.loader.load_raw_metadata()
        
        # 1.1 Perform pre-extraction metadata-level filtering to optimize memory
        filtered_metadata_df, skipped_metadata_list = self.loader.filter_metadata_before_extraction(raw_df)
        
        # 1.2 Extract sequences only for the filtered subset
        valid_loaded_df, skipped_extraction_df = self.loader.extract_dataset(filtered_metadata_df, limit=limit)
        
        # 1.3 Compile loader skips
        skipped_loader_df = pd.concat([
            pd.DataFrame(skipped_metadata_list),
            skipped_extraction_df
        ], ignore_index=True)
        
        # Save loading skips report
        skipped_loader_path = os.path.join(self.config.output_dir, "eccDNA_skipped_loader.csv")
        skipped_loader_df.to_csv(skipped_loader_path, index=False)
        logger.info(f"Loader skips saved to {skipped_loader_path}")

        if valid_loaded_df.empty:
            logger.error("No valid sequences extracted. Terminating pipeline.")
            return pd.DataFrame(), pd.DataFrame()

        # --------------------------------------------------
        # Step 2: Data Cleaning
        # --------------------------------------------------
        logger.info("--- STAGE 2: Data Cleaning & Deduplication ---")
        cleaned_df, discarded_cleaner_df = self.cleaner.clean(valid_loaded_df)
        
        # Save cleaning discards report
        discarded_cleaner_path = os.path.join(self.config.output_dir, "eccDNA_discarded_cleaner.csv")
        discarded_cleaner_df.to_csv(discarded_cleaner_path, index=False)
        logger.info(f"Cleaner discards saved to {discarded_cleaner_path}")

        if cleaned_df.empty:
            logger.error("No sequences remained after cleaning. Terminating pipeline.")
            return pd.DataFrame(), pd.DataFrame()

        # --------------------------------------------------
        # Step 3: Disease Filtering
        # --------------------------------------------------
        logger.info("--- STAGE 3: Disease Filtering ---")
        filtered_df, filter_dict, filter_md = self.disease_filter.filter_and_report(cleaned_df)
        
        # Save disease filtering report
        report_dir = os.path.join(self.config.output_dir, "reports")
        report_path = os.path.join(report_dir, "disease_filtering_report.md")
        save_report(filter_md, report_path)

        if filtered_df.empty:
            logger.error("No sequences remained after disease filtering. Terminating pipeline.")
            return pd.DataFrame(), pd.DataFrame()

        # --------------------------------------------------
        # Step 4: Sequence Feature Extraction
        # --------------------------------------------------
        logger.info("--- STAGE 4: Sequence Feature Extraction ---")
        logger.info(f"Extracting features for {len(filtered_df)} sequences...")
        
        feature_records = []
        for idx, row in filtered_df.iterrows():
            seq = row["sequence"]
            sample_id = row["sample_id"]
            genomic_len = row["genomic_length"]
            
            # Extract features
            features = self.extractor.extract_all_features(
                seq, 
                sample_id, 
                genomic_len, 
                include_kmers=include_kmers
            )
            
            # Merge with existing clinical/experimental metadata columns
            metadata_fields = {
                "disease": row["disease"],
                "species": row["species"],
                "chromosome": row["chromosome"],
                "start": row["start"],
                "end": row["end"],
                "genomic_coordinates": row["genomic_coordinates"],
                "tissue": row["tissue"],
                "sample_source": row["sample_source"],
                "eccdna_type": row["eccdna_type"],
                "health_status": row["health_status"],
                "library_type": row["library_type"],
                "pubmed_id": row["pubmed_id"],
                "public_date": row["public_date"],
                "experiment_prediction": row["experiment_prediction"],
                "validation_strategies": row["validation_strategies"],
                "treatment": row["treatment"],
                "oncogene": row["oncogene"],
                "function_characteristic": row["function_characteristic"],
                "remarks": row["remarks"],
                "sequence": row["sequence"]  # Keep raw sequence for alignment-free verification
            }
            features.update(metadata_fields)
            feature_records.append(features)
            
            if (len(feature_records)) % 50000 == 0:
                logger.info(f"Features extracted: {len(feature_records)} / {len(filtered_df)}")

        df_features = pd.DataFrame(feature_records)
        logger.info(f"Feature matrix constructed with shape {df_features.shape}")

        # --------------------------------------------------
        # Step 5: Disease-level Statistical Analysis
        # --------------------------------------------------
        logger.info("--- STAGE 5: Disease-level Statistical Aggregation ---")
        summary_df = self.stats_aggregator.aggregate_statistics(df_features)
        summary_md = self.stats_aggregator.generate_publication_table(summary_df)
        
        # Save statistics report
        stats_path = os.path.join(report_dir, "feature_statistics_summary.md")
        save_report(summary_md, stats_path)

        # --------------------------------------------------
        # Step 6: Exploratory Plots & Outlier Detection
        # --------------------------------------------------
        logger.info("--- STAGE 6: Visualization and Outlier Filtering ---")
        plots_dir = os.path.join(self.config.output_dir, "plots")
        
        # Plot Distributions (Length and GC)
        dists_plot_path = os.path.join(plots_dir, "distributions.png")
        self.visualizer.plot_distributions(df_features, dists_plot_path)
        
        # Plot Average MI profiles
        mi_plot_path = os.path.join(plots_dir, "mi_profiles.png")
        self.visualizer.plot_mi_profiles(df_features, mi_plot_path)
        
        # Plot Projections (PCA / UMAP / t-SNE)
        proj_plot_path = os.path.join(plots_dir, "projections.png")
        self.visualizer.plot_projections(df_features, proj_plot_path)

        # Detect Outliers (Isolation Forest)
        normal_features_df, outlier_features_df = self.visualizer.detect_outliers(df_features)

        # Save Final clean and outlier feature datasets
        clean_features_path = os.path.join(self.config.output_dir, "eccDNA_sequences_clean.csv")
        normal_features_df.to_csv(clean_features_path, index=False)
        
        outliers_path = os.path.join(self.config.output_dir, "eccDNA_outliers.csv")
        outlier_features_df.to_csv(outliers_path, index=False)
        
        logger.info(f"Final clean features saved to {clean_features_path} ({len(normal_features_df)} records)")
        logger.info(f"Isolated outliers saved to {outliers_path} ({len(outlier_features_df)} records)")
        logger.info("==================================================")
        logger.info("PREPROCESSING PIPELINE EXECUTION COMPLETED")
        logger.info("==================================================")

        return normal_features_df, outlier_features_df
