# Pinecone Configuration
PINECONE_API_KEY = "pcsk_2zy16Y_2WcDUdg6KW8hdkis695fJikuVMtQETk3YXie3SDxfUSXNpFw2wjgsRNWv6fTsEh"

# Index Configuration
INDEX_NAME = "ecommerce-products"
DIMENSION = 384  # Using sentence-transformers all-MiniLM-L6-v2 (384 dimensions)
METRIC = "cosine"  # Similarity metric

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"