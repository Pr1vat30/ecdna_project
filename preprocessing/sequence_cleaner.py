import logging
from typing import Tuple, Dict, Any
import pandas as pd
from preprocessing.config import PreprocessingConfig

logger = logging.getLogger(__name__)

class SequenceCleaner:
    """
    Step 2: Data Cleaning
    Redesigns the sequence cleaning pipeline. Checks for coordinate/sequence duplicates,
    standardizes casing, filters out short/long sequences, and flags ambiguous IUPAC nucleotides.
    """
    def __init__(self, config: PreprocessingConfig, max_ambiguous_fraction: float = 0.01):
        self.config = config
        self.max_ambiguous_fraction = max_ambiguous_fraction
        self.standard_bases = {'A', 'C', 'G', 'T'}

    def is_ambiguous_fraction_valid(self, seq: str) -> bool:
        """
        Calculates the fraction of non-standard bases in the sequence.
        Returns True if the fraction is within acceptable limits, False otherwise.
        """
        if not seq:
            return False
            
        non_standard_count = sum(1 for base in seq if base not in self.standard_bases)
        fraction = non_standard_count / len(seq)
        return fraction <= self.max_ambiguous_fraction

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies cleaning steps to the extracted DataFrame.
        Returns a tuple (cleaned_df, discarded_df).
        """
        if df.empty:
            logger.warning("Input DataFrame to cleaner is empty.")
            return df, pd.DataFrame(columns=df.columns.tolist() + ["discard_reason"])

        logger.info(f"Starting data cleaning on {len(df)} records...")
        
        # Keep track of discarded records
        discarded_records = []

        # 1. Convert sequence to uppercase and validate coordinates
        df_processed = df.copy()
        df_processed["sequence"] = df_processed["sequence"].str.upper()

        cleaned_rows = []
        
        for idx, row in df_processed.iterrows():
            seq = row["sequence"]
            seq_len = len(seq)
            start = row["start"]
            end = row["end"]
            id_val = row["sample_id"]
            
            # 1.1 Sequence length check
            if seq_len < self.config.min_seq_len:
                discarded_records.append({
                    **row.to_dict(),
                    "discard_reason": f"Sequence length {seq_len} < min threshold {self.config.min_seq_len}"
                })
                continue
                
            if seq_len > self.config.max_seq_len:
                discarded_records.append({
                    **row.to_dict(),
                    "discard_reason": f"Sequence length {seq_len} > max threshold {self.config.max_seq_len}"
                })
                continue

            # 1.2 Coordinate sanity check (already partially filtered in loader, but good as safety)
            if start >= end:
                discarded_records.append({
                    **row.to_dict(),
                    "discard_reason": f"Invalid coordinates: start ({start}) >= end ({end})"
                })
                continue

            # 1.3 Ambiguous nucleotide fraction check
            if not self.is_ambiguous_fraction_valid(seq):
                discarded_records.append({
                    **row.to_dict(),
                    "discard_reason": "Ambiguous nucleotide fraction > 1%"
                })
                continue

            cleaned_rows.append(row)

        df_filtered = pd.DataFrame(cleaned_rows) if cleaned_rows else pd.DataFrame(columns=df.columns)
        
        if df_filtered.empty:
            logger.warning("All records were discarded during coordinate/length/ambiguity filtering.")
            return df_filtered, pd.DataFrame(discarded_records)

        # 2. Hashing-based Deduplication to prevent ML Data Leakage
        # 2.1 Coordinate deduplication (same chromosome, start, end)
        initial_len = len(df_filtered)
        coord_dup_mask = df_filtered.duplicated(subset=["chromosome", "start", "end"], keep="first")
        df_coord_dedup = df_filtered[~coord_dup_mask].copy()
        
        coord_duplicates = df_filtered[coord_dup_mask]
        for _, row in coord_duplicates.iterrows():
            discarded_records.append({
                **row.to_dict(),
                "discard_reason": f"Duplicate genomic coordinates ({row['chromosome']}:{row['start']}-{row['end']})"
            })
            
        logger.info(f"Coordinate deduplication: removed {len(coord_duplicates)} duplicate coordinate records.")

        # 2.2 Sequence deduplication (identical DNA sequences)
        initial_seq_len = len(df_coord_dedup)
        seq_dup_mask = df_coord_dedup.duplicated(subset=["sequence"], keep="first")
        df_cleaned = df_coord_dedup[~seq_dup_mask].copy()
        
        seq_duplicates = df_coord_dedup[seq_dup_mask]
        for _, row in seq_duplicates.iterrows():
            discarded_records.append({
                **row.to_dict(),
                "discard_reason": "Duplicate DNA sequence"
            })
            
        logger.info(f"Sequence deduplication: removed {len(seq_duplicates)} duplicate DNA sequence records.")

        discarded_df = pd.DataFrame(discarded_records) if discarded_records else pd.DataFrame(columns=df.columns.tolist() + ["discard_reason"])
        
        logger.info(f"Cleaning completed. Remaining: {len(df_cleaned)} records. Discarded: {len(discarded_df)} records.")
        return df_cleaned, discarded_df
