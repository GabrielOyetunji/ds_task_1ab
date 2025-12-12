"""
Data Cleaning Service for E-commerce Dataset

Handles cleaning and preprocessing of raw e-commerce data including
text normalization, duplicate removal, and missing value handling.
"""

import pandas as pd
import os
import re
import logging
from typing import Tuple, Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and preprocesses e-commerce dataset for analysis and vectorization."""
    
    def __init__(
        self,
        main_dataset_path: str = "data/raw/dataset.csv",
        cnn_codes_path: str = "data/raw/CNN_Model_Train_Data.csv",
        output_dir: str = "data/processed"
    ) -> None:
        """Initialize data cleaner with file paths.
        
        Args:
            main_dataset_path: Path to main dataset CSV
            cnn_codes_path: Path to CNN training codes CSV
            output_dir: Directory to save cleaned data
        """
        self.main_dataset_path = main_dataset_path
        self.cnn_codes_path = cnn_codes_path
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"DataCleaner initialized. Output: {output_dir}")

    def clean_main_dataset(self) -> pd.DataFrame:
        """Clean and preprocess main e-commerce dataset.
        
        Performs the following operations:
        - Removes invalid characters from InvoiceNo and StockCode
        - Cleans Description text
        - Normalizes Country names
        - Removes rows with missing critical fields
        - Filters invalid quantities and prices
        
        Returns:
            Cleaned pandas DataFrame
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset is empty after cleaning
        """
        try:
            if not os.path.exists(self.main_dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.main_dataset_path}")
            
            logger.info(f"Loading dataset from {self.main_dataset_path}")
            df = pd.read_csv(self.main_dataset_path, encoding="latin1")
            initial_rows = len(df)
            logger.info(f"Initial rows: {initial_rows}")

            # Clean InvoiceNo
            logger.info("Cleaning InvoiceNo...")
            df["InvoiceNo"] = (
                df["InvoiceNo"].astype(str)
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .str.strip()
            )

            # Clean StockCode
            logger.info("Cleaning StockCode...")
            df["StockCode"] = (
                df["StockCode"].astype(str)
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .str.strip()
            )

            # Clean Description
            logger.info("Cleaning Description...")
            df["Description"] = (
                df["Description"].astype(str)
                .str.replace(r"[^\w\s]", "", regex=True)
                .str.strip()
            )

            # Clean Quantity
            logger.info("Cleaning Quantity...")
            df["Quantity"] = pd.to_numeric(
                df["Quantity"].astype(str).str.replace(r"[^\d]", "", regex=True),
                errors="coerce"
            )

            # Clean UnitPrice
            logger.info("Cleaning UnitPrice...")
            df["UnitPrice"] = pd.to_numeric(
                df["UnitPrice"], errors="coerce"
            )

            # Clean Country
            logger.info("Cleaning Country...")
            df["Country"] = (
                df["Country"].astype(str)
                .str.replace(r"[^A-Za-z\s]", "", regex=True)
                .str.strip()
            )
            df["Country"] = df["Country"].str.replace(
                r"^[A-Za-z]{1,4}(?=[A-Z])", "", regex=True
            )

            # Remove rows with empty critical fields
            logger.info("Removing rows with missing critical fields...")
            before_filter = len(df)
            df = df[df["InvoiceNo"] != ""]
            df = df[df["StockCode"] != ""]
            df = df[df["Description"] != ""]
            df = df[df["Country"] != ""]
            logger.info(f"Removed {before_filter - len(df)} rows with empty fields")

            # Remove invalid quantities and prices
            logger.info("Filtering valid quantities and prices...")
            before_filter = len(df)
            df = df[df["Quantity"] > 0]
            df = df[df["UnitPrice"] > 0]
            logger.info(f"Removed {before_filter - len(df)} rows with invalid values")

            # Remove duplicates
            logger.info("Removing duplicates...")
            before_dedup = len(df)
            df = df.drop_duplicates()
            logger.info(f"Removed {before_dedup - len(df)} duplicate rows")

            final_rows = len(df)
            if final_rows == 0:
                raise ValueError("Dataset is empty after cleaning")
            
            logger.info(f"Cleaning complete: {initial_rows} → {final_rows} rows")
            logger.info(f"Retention rate: {final_rows/initial_rows*100:.1f}%")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning main dataset: {e}")
            raise

    def clean_cnn_codes(self) -> pd.DataFrame:
        """Clean CNN training codes dataset.
        
        Returns:
            Cleaned pandas DataFrame with stock codes
            
        Raises:
            FileNotFoundError: If CNN codes file doesn't exist
        """
        try:
            if not os.path.exists(self.cnn_codes_path):
                raise FileNotFoundError(f"CNN codes not found: {self.cnn_codes_path}")
            
            logger.info(f"Loading CNN codes from {self.cnn_codes_path}")
            df = pd.read_csv(self.cnn_codes_path)
            initial_rows = len(df)
            
            logger.info("Cleaning StockCode column...")
            df["StockCode"] = (
                df["StockCode"].astype(str)
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .str.strip()
            )
            
            df = df[df["StockCode"] != ""]
            df = df.drop_duplicates(subset=["StockCode"])
            
            final_rows = len(df)
            logger.info(f"CNN codes cleaned: {initial_rows} → {final_rows} unique codes")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning CNN codes: {e}")
            raise

    def save_cleaned_data(
        self,
        main_df: pd.DataFrame,
        cnn_df: pd.DataFrame
    ) -> Tuple[str, str]:
        """Save cleaned datasets to output directory.
        
        Args:
            main_df: Cleaned main dataset
            cnn_df: Cleaned CNN codes dataset
            
        Returns:
            Tuple of (main_output_path, cnn_output_path)
        """
        try:
            main_output = os.path.join(self.output_dir, "cleaned_data.csv")
            cnn_output = os.path.join(self.output_dir, "cnn_train_codes_clean.csv")
            
            logger.info(f"Saving cleaned main dataset to {main_output}")
            main_df.to_csv(main_output, index=False)
            
            logger.info(f"Saving cleaned CNN codes to {cnn_output}")
            cnn_df.to_csv(cnn_output, index=False)
            
            logger.info("All cleaned data saved successfully")
            return main_output, cnn_output
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise

    def run_cleaning_pipeline(self) -> Tuple[str, str]:
        """Execute full data cleaning pipeline.
        
        Returns:
            Tuple of (main_output_path, cnn_output_path)
            
        Raises:
            RuntimeError: If cleaning pipeline fails
        """
        try:
            logger.info("="*50)
            logger.info("STARTING DATA CLEANING PIPELINE")
            logger.info("="*50)
            
            main_df = self.clean_main_dataset()
            cnn_df = self.clean_cnn_codes()
            paths = self.save_cleaned_data(main_df, cnn_df)
            
            logger.info("="*50)
            logger.info("DATA CLEANING COMPLETE")
            logger.info(f"Main dataset: {paths[0]}")
            logger.info(f"CNN codes: {paths[1]}")
            logger.info("="*50)
            
            return paths
            
        except Exception as e:
            logger.error(f"Cleaning pipeline failed: {e}")
            raise RuntimeError(f"Data cleaning failed: {e}")


if __name__ == "__main__":
    try:
        cleaner = DataCleaner()
        main_path, cnn_path = cleaner.run_cleaning_pipeline()
        logger.info("Task 1 completed successfully")
    except Exception as e:
        logger.error(f"Task 1 failed: {e}")
        raise
