import pandas as pd
import os
import re


class DataCleaner:
    def __init__(self,
                 main_dataset_path="data/raw/dataset.csv",
                 cnn_codes_path="data/raw/CNN_Model_Train_Data.csv",
                 output_dir="data/processed"):

        self.main_dataset_path = main_dataset_path
        self.cnn_codes_path = cnn_codes_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def clean_main_dataset(self):
        df = pd.read_csv(self.main_dataset_path, encoding="latin1")

        # Clean InvoiceNo
        df["InvoiceNo"] = (
            df["InvoiceNo"].astype(str)
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.strip()
        )

        # Clean StockCode
        df["StockCode"] = (
            df["StockCode"].astype(str)
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.strip()
        )

        # Clean Description
        df["Description"] = (
            df["Description"].astype(str)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Clean CustomerID
        df["CustomerID"] = (
            df["CustomerID"].astype(str)
            .str.replace(r"[^\d]", "", regex=True)
        )
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

        # Clean UnitPrice
        df["UnitPrice"] = (
            df["UnitPrice"].astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

        # Clean Quantity
        df["Quantity"] = (
            df["Quantity"].astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

        # FIX COUNTRY COLUMN
        # Remove all non-letter characters first
        df["Country"] = (
            df["Country"].astype(str)
            .str.replace(r"[^A-Za-z\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Remove strange prefixes like XxY, XX, YY, etc.
        df["Country"] = df["Country"].str.replace(r"^[A-Za-z]{1,4}(?=[A-Z])", "", regex=True).str.strip()

        # Keep only known country names (optional but improves quality)
        df = df[df["Country"].str.len() > 0]

        # Remove missing values
        df = df.dropna(subset=["InvoiceNo", "StockCode", "Description", "CustomerID", "UnitPrice", "Quantity"])

        # Remove empty strings
        df = df[df["InvoiceNo"].str.len() > 0]
        df = df[df["StockCode"].str.len() > 0]
        df = df[df["Description"].str.len() > 0]

        # Remove duplicates
        df = df.drop_duplicates()

        # Filter invalid prices and quantities
        df = df[df["UnitPrice"] > 0]
        df = df[df["Quantity"] > 0]

        output_path = os.path.join(self.output_dir, "cleaned_data.csv")
        df.to_csv(output_path, index=False)

        print(f"Cleaned dataset: {len(df)} rows")
        return output_path

    def clean_cnn_codes(self):
        df = pd.read_csv(self.cnn_codes_path, encoding="latin1")

        df["StockCode"] = (
            df["StockCode"].astype(str)
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .str.strip()
        )

        df = df[df["StockCode"].str.len() > 0]

        output_path = os.path.join(self.output_dir, "cnn_train_codes_clean.csv")
        df.to_csv(output_path, index=False)

        print(f"Cleaned CNN codes: {len(df)} codes")
        return output_path


if __name__ == "__main__":
    cleaner = DataCleaner()

    print("Cleaning main dataset...")
    main_path = cleaner.clean_main_dataset()
    print("Saved:", main_path)

    print("\nCleaning CNN train codes...")
    cnn_path = cleaner.clean_cnn_codes()
    print("Saved:", cnn_path)

    print("\nAll cleaning completed successfully.")