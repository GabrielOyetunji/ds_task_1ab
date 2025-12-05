# Module 1 Report  
### Data Preparation and Backend Setup

## Introduction  
This module prepared the product dataset for downstream processing. Tasks included cleaning, vector creation, establishing a vector index, and building the base recommendation service.

## High-Level Flow  
1. Clean raw product data  
2. Standardize formats, remove duplicates  
3. Generate embeddings using MiniLM  
4. Store vectors in Pinecone  
5. Query vectors to retrieve similar products

## Description of Work  
- Cleaned the dataset (missing values removed, formats normalized)  
- Selected MiniLM for embeddings  
- Connected Pinecone index named "ecommerce-products"  
- Implemented vector insert and search operations  
- Built the recommendation endpoint that uses both embedding search and text normalisation

## Key Decisions  
- Chose cosine similarity since it provides stable rankings for text-based embeddings  
- Defined schema with product description, price, and stock code  
- Added input checks to prevent harmful or poor queries

## Challenges and Solutions  
- Inconsistent product names solved by standard tokenisation  
- Several duplicate records removed before vectorization  
- Early index mismatches solved by ensuring same dimensionality across embeddings  

## Conclusion  
Module 1 delivered a working recommendation engine using vector search. The cleaned dataset now supports all later modules.

## References  
- SentenceTransformers MiniLM-L6-v2  
- Pinecone Python SDK  
- Flask Framework
