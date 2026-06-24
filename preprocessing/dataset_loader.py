import os
import logging
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from pyfaidx import Fasta
from preprocessing.config import PreprocessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Step 1: Dataset Construction
    Loads the raw metadata from CSV and extracts genomic sequences from the reference genome.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._genome: Optional[Fasta] = None

    @property
    def genome(self) -> Fasta:
        """Lazy load the reference genome to save startup time and memory."""
        if self._genome is None:
            if not os.path.exists(self.config.reference_genome_path):
                raise FileNotFoundError(f"Reference genome not found at {self.config.reference_genome_path}")
            logger.info(f"Loading reference genome from {self.config.reference_genome_path}...")
            self._genome = Fasta(self.config.reference_genome_path)
            logger.info("Reference genome loaded successfully.")
        return self._genome

    def load_raw_metadata(self) -> pd.DataFrame:
        """
        Loads the raw metadata CSV file.
        Uses low_memory=False to prevent dtype warnings for mixed types in PubMed ID and Public Date.
        """
        if not os.path.exists(self.config.raw_metadata_path):
            raise FileNotFoundError(f"Raw metadata CSV not found at {self.config.raw_metadata_path}")
        logger.info(f"Loading raw metadata from {self.config.raw_metadata_path}...")
        df = pd.read_csv(self.config.raw_metadata_path, low_memory=False)
        logger.info(f"Raw metadata loaded. Found {len(df)} records.")
        return df

    def parse_coordinates(self, row: pd.Series) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
        """
        Parses chromosome, start, and end coordinates from a metadata row.
        Handles coordinate retrieval from Chr, Start, End fields.
        Returns (chrom, start, end, error_msg).
        """
        # Read fields
        chrom = row.get("Chr")
        start_val = row.get("Start")
        end_val = row.get("End")
        
        # Check for missing values
        if pd.isna(chrom) or pd.isna(start_val) or pd.isna(end_val):
            return None, None, None, "Missing chromosome or coordinate values"
            
        chrom = str(chrom).strip()
        if chrom == "Not Available" or chrom == "":
            return None, None, None, "Chromosome name 'Not Available' or empty"
            
        try:
            start = int(float(start_val))
            end = int(float(end_val))
        except (ValueError, TypeError):
            return None, None, None, f"Invalid coordinate format (Start: {start_val}, End: {end_val})"
            
        if start <= 0 or end <= 0:
            return None, None, None, f"Coordinates must be positive integers (Start: {start}, End: {end})"
            
        if start > end:
            return None, None, None, f"Start coordinate greater than End coordinate (Start: {start}, End: {end})"
            
        return chrom, start, end, None

    def filter_metadata_before_extraction(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Filters raw metadata at the coordinate and class level BEFORE sequence extraction
        to optimize memory and prevent SIGKILL.
        """
        logger.info("Performing pre-extraction filtering on raw metadata...")
        skipped = []
        valid_indices = []
        
        # Find coordinates errors and record skips
        for idx, row in df.iterrows():
            id_val = row.get("eccDNA ID", f"Row_{idx}")
            chrom, start, end, coord_error = self.parse_coordinates(row)
            if coord_error:
                skipped.append({
                    "id": id_val,
                    "reason": coord_error,
                    "row_index": idx
                })
            else:
                valid_indices.append(idx)
                
        df_valid = df.loc[valid_indices].copy()
        logger.info(f"Metadata coordinate parsing completed. Valid coords: {len(df_valid)}, Skipped: {len(skipped)}")
        
        # Safe coordinate casting
        df_valid["Start"] = df_valid["Start"].astype(float).astype(int)
        df_valid["End"] = df_valid["End"].astype(float).astype(int)
        df_valid["Chr"] = df_valid["Chr"].astype(str).str.strip()
        
        # 1. Coordinate deduplication
        before_dedup = len(df_valid)
        df_valid = df_valid.drop_duplicates(subset=["Chr", "Start", "End"], keep="first")
        logger.info(f"Metadata coordinate deduplication: removed {before_dedup - len(df_valid)} duplicate coordinate records.")
        
        # 2. Exclude 'Not Available' diseases
        df_valid = df_valid[df_valid["Disease Name"] != "Not Available"]
        
        # 3. Disease sample size filtering
        disease_counts = df_valid["Disease Name"].value_counts()
        valid_diseases = disease_counts[disease_counts >= self.config.min_disease_samples].index
        df_valid = df_valid[df_valid["Disease Name"].isin(valid_diseases)].copy()
        logger.info(f"Metadata disease filtering: retained {df_valid['Disease Name'].nunique()} diseases.")
        
        # 4. Class balance sampling (max_per_class)
        # Using explicit loop to avoid pandas version groupby-apply index issues
        sampled_dfs = []
        for disease, group in df_valid.groupby("Disease Name"):
            sampled_group = group.sample(n=min(len(group), self.config.max_per_class), random_state=42)
            sampled_dfs.append(sampled_group)
        if sampled_dfs:
            df_valid = pd.concat(sampled_dfs, ignore_index=True)
        else:
            df_valid = pd.DataFrame(columns=df.columns)
        
        logger.info(f"Pre-extraction filtering completed. Retained {len(df_valid)} records for sequence extraction.")
        return df_valid, skipped

    def extract_dataset(self, df: pd.DataFrame, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts DNA sequences and compiles metadata for each row in df.
        Does not perform disease filtering and sampling inside this method (which is done at metadata level).
        Returns a tuple of (valid_records_df, skipped_records_df).
        """
        records = []
        skipped = []
        
        # Initialize genome to ensure we catch missing genome file errors early
        _ = self.genome
        
        total_rows = len(df) if limit is None else min(limit, len(df))
        logger.info(f"Starting sequence extraction for {total_rows} rows...")
        
        df_subset = df.head(total_rows)
        
        for idx, row in df_subset.iterrows():
            id_val = row.get("eccDNA ID", f"Row_{idx}")
            disease = row.get("Disease Name", "Not Available")
            
            # Coordinate Parsing
            chrom, start, end, coord_error = self.parse_coordinates(row)
            if coord_error:
                skipped.append({
                    "id": id_val,
                    "reason": coord_error,
                    "row_index": idx
                })
                continue
            
            # Sequence Extraction (1-based closed coordinate convention)
            try:
                if chrom not in self.genome:
                    skipped.append({
                        "id": id_val,
                        "reason": f"Chromosome {chrom} not found in reference genome",
                        "row_index": idx
                    })
                    continue
                
                # Slicing is 0-indexed in python, start is 1-based, so start-1 is correct.
                # End is 1-based closed, python slice is exclusive, so genome[chrom][start-1:end] yields end-start+1 bases.
                seq_obj = self.genome[chrom][start - 1:end]
                seq = seq_obj.seq.upper()
                
            except Exception as e:
                skipped.append({
                    "id": id_val,
                    "reason": f"Fasta extraction error: {str(e)}",
                    "row_index": idx
                })
                continue
                
            # Species name standardization
            species = str(row.get("Species", "Homo sapiens")).strip()
            # Standardize case (e.g. 'Homo sapiens' vs 'Homo Sapiens')
            if species.lower() == "homo sapiens":
                species = "Homo sapiens"
            
            # Compile metadata dictionary
            record = {
                # Primary identity and sequence
                "sample_id": id_val,
                "sequence": seq,
                "disease": str(disease).strip(),
                "species": species,
                
                # Coordinates
                "chromosome": chrom,
                "start": start,
                "end": end,
                "genomic_coordinates": str(row.get("Location", f"{chrom}:{start}-{end}")).strip(),
                
                # Lengths
                "sequence_length": len(seq),
                "genomic_length": end - start + 1,
                
                # Biological/Experimental metadata
                "tissue": str(row.get("Tissue/Cell line", "Not Available")).strip(),
                "sample_source": str(row.get("Source", "Not Available")).strip(),
                "eccdna_type": str(row.get("eccDNA type", "Not Available")).strip(),
                "health_status": str(row.get("Health/Disease", "Not Available")).strip(),
                "library_type": str(row.get("Library type", "Not Available")).strip(),
                "pubmed_id": str(row.get("Pubmed ID", "Not Available")).strip(),
                "public_date": str(row.get("Public Date", "Not Available")).strip(),
                
                # Additional experimental details
                "experiment_prediction": str(row.get("Experiment/Prediction", "Not Available")).strip(),
                "validation_strategies": str(row.get("Validation Strategies", "Not Available")).strip(),
                "treatment": str(row.get("Treatment", "Not Available")).strip(),
                "oncogene": str(row.get("Oncogene", "Not Available")).strip(),
                "function_characteristic": str(row.get("Function/Characteristic of ecDNA", "Not Available")).strip(),
                "remarks": str(row.get("Remarks", "Not Available")).strip()
            }
            
            records.append(record)
            
            if (len(records)) % 20000 == 0:
                logger.info(f"Processed {len(records)} / {total_rows} rows...")
                
        logger.info(f"Extraction completed. Valid records: {len(records)}, Skipped: {len(skipped)}")
        
        valid_df = pd.DataFrame(records)
        skipped_df = pd.DataFrame(skipped)
        
        return valid_df, skipped_df
