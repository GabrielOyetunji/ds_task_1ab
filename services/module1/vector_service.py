"""
Vector Database Service for Product Embeddings

Manages product embeddings using Pinecone vector database and
sentence-transformers for semantic search.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time
import sys
import os
import logging
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PINECONE_API_KEY, INDEX_NAME, DIMENSION, METRIC, EMBEDDING_MODEL


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorService:
    """Manages vector embeddings and Pinecone database operations."""
    
    def __init__(self) -> None:
        """Initialize Pinecone connection and embedding model."""
        try:
            logger.info("Initializing VectorService...")
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.index = None
            logger.info("VectorService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VectorService: {e}")
            raise

    def create_index(self, delete_if_exists: bool = False) -> None:
        """Create or connect to Pinecone index.
        
        Args:
            delete_if_exists: If True, delete existing index before creating new one
            
        Raises:
            RuntimeError: If index creation fails
        """
        try:
            logger.info(f"Setting up index: {INDEX_NAME}")
            existing_indexes = [i.name for i in self.pc.list_indexes()]

            if INDEX_NAME in existing_indexes:
                if delete_if_exists:
                    logger.warning(f"Deleting existing index: {INDEX_NAME}")
                    self.pc.delete_index(INDEX_NAME)
                    time.sleep(1)
                else:
                    logger.info(f"Index exists. Connecting to {INDEX_NAME}")
                    self.index = self.pc.Index(INDEX_NAME)
                    return

            logger.info(f"Creating new index: dimension={DIMENSION}, metric={METRIC}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            logger.info("Waiting for index to become ready...")
            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

            self.index = self.pc.Index(INDEX_NAME)
            logger.info(f"Index {INDEX_NAME} is ready")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise RuntimeError(f"Failed to create index: {e}")

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for text descriptions.
        
        Args:
            texts: List of text descriptions to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
        """
        try:
            if not texts:
                raise ValueError("Cannot generate embeddings for empty text list")
            
            logger.info(f"Generating embeddings for {len(texts)} items...")
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info("Embeddings generated successfully")
            return vectors.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def upload_products(
        self,
        csv_path: str,
        batch_size: int = 100
    ) -> None:
        """Upload product vectors to Pinecone.
        
        Args:
            csv_path: Path to cleaned product data CSV
            batch_size: Number of vectors to upload per batch
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is empty or missing required columns
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            logger.info(f"Loading cleaned data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            logger.info(f"Total rows: {len(df)}")

            logger.info("Extracting unique products by StockCode...")
            unique_products = df.groupby("StockCode").agg({
                "Description": "first",
                "UnitPrice": "mean",
                "Country": "first"
            }).reset_index()

            logger.info(f"Unique products found: {len(unique_products)}")

            descriptions = unique_products["Description"].tolist()
            embeddings = self.generate_embeddings(descriptions)

            logger.info("Preparing vectors for upload...")
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

            logger.info(f"Uploading {len(vectors)} vectors...")
            for i in tqdm(range(0, len(vectors), batch_size)):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            stats = self.index.describe_index_stats()
            logger.info(f"Upload complete. Total vectors: {stats['total_vector_count']}")
            
        except Exception as e:
            logger.error(f"Error uploading products: {e}")
            raise

    def search_similar_products(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar products using natural language query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            List of product dictionaries with metadata and scores
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search fails
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if self.index is None:
                raise RuntimeError("Index not initialized. Call create_index() first")
            
            logger.info(f"Searching for: '{query}' (top {top_k})")
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
            
            logger.info(f"Found {len(products)} matching products")
            return products
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            raise


if __name__ == "__main__":
    try:
        service = VectorService()
        service.create_index(delete_if_exists=False)
        service.upload_products("data/processed/cleaned_data.csv")
        logger.info("Task 2 completed successfully")
    except Exception as e:
        logger.error(f"Task 2 failed: {e}")
        raise
