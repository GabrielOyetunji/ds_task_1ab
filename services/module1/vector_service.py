import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PINECONE_API_KEY, INDEX_NAME, DIMENSION, METRIC, EMBEDDING_MODEL


class VectorService:
    def __init__(self):
        print("Initializing VectorService...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        print("VectorService initialized.")

    def create_index(self, delete_if_exists=False):
        print(f"Creating index: {INDEX_NAME}")
        existing_indexes = [i.name for i in self.pc.list_indexes()]

        if INDEX_NAME in existing_indexes:
            if delete_if_exists:
                print(f"Deleting existing index: {INDEX_NAME}")
                self.pc.delete_index(INDEX_NAME)
                time.sleep(1)
            else:
                print(f"Index exists. Connecting to {INDEX_NAME}.")
                self.index = self.pc.Index(INDEX_NAME)
                return

        print(f"Creating new index with dimension={DIMENSION}, metric={METRIC}")
        self.pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Waiting for index to become ready...")
        while not self.pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

        self.index = self.pc.Index(INDEX_NAME)
        print(f"Index {INDEX_NAME} is ready.")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        print(f"Generating embeddings for {len(texts)} items...")
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return vectors.tolist()

    def upload_products(self, csv_path: str, batch_size: int = 100):
        print(f"Loading cleaned data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Total rows: {len(df)}")

        print("Extracting unique products by StockCode...")
        unique_products = df.groupby("StockCode").agg({
            "Description": "first",
            "UnitPrice": "mean",
            "Country": "first"
        }).reset_index()

        print(f"Unique products found: {len(unique_products)}")

        descriptions = unique_products["Description"].tolist()
        embeddings = self.generate_embeddings(descriptions)

        print("Preparing vectors for upload...")
        vectors = []
        for idx, row in unique_products.iterrows():
            vector_id = f"product_{row['StockCode']}"
            metadata = {
                "stock_code": str(row["StockCode"]),
                "description": str(row["Description"]),
                "price": float(row["UnitPrice"]),
                "country": str(row["Country"])
            }
            vectors.append((vector_id, embeddings[idx], metadata))

        print(f"Uploading {len(vectors)} vectors...")
        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        stats = self.index.describe_index_stats()
        print(f"Upload complete. Total vectors stored: {stats['total_vector_count']}")

    def search_similar_products(self, query, top_k=5):
        """
        Search for similar products using natural language query.
        """
        query_embedding = self.model.encode(query).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        products = []
        for match in results['matches']:
            products.append({
                'stock_code': match['metadata']['stock_code'],
                'description': match['metadata']['description'],
                'price': match['metadata']['price'],
                'country': match['metadata']['country'],
                'similarity_score': round(match['score'], 4)
            })
        
        return products


if __name__ == "__main__":
    service = VectorService()
    service.create_index(delete_if_exists=False)
    service.upload_products("data/processed/cleaned_data.csv")
    print("Task 2 completed.")