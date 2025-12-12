# E-commerce Product Recommendation System

A comprehensive AI-powered e-commerce platform featuring product recommendations, OCR-based query processing, and CNN-based product detection.

## Project Overview

This system provides three main functionalities:
1. **Text-based Product Recommendations** - Natural language search using vector similarity
2. **OCR Query Processing** - Handwritten text extraction and product search
3. **Image-based Product Detection** - CNN model for product classification

## Technology Stack

- **Backend**: Flask
- **ML/AI**: PyTorch, sentence-transformers, Donut OCR
- **Vector Database**: Pinecone
- **Image Processing**: torchvision, PIL
- **Data Processing**: pandas, numpy

## Project Structure
```
ds_task_1ab/
├── services/
│   ├── module1/          # Data cleaning and vector search
│   │   ├── data_cleaner.py
│   │   ├── vector_service.py
│   │   └── recommendation_service.py
│   ├── module2/          # OCR and web scraping
│   │   ├── ocr_service.py
│   │   └── scraper.py
│   └── module3/          # CNN model training and detection
│       ├── cnn_service.py
│       └── detection_service.py
├── notebooks/            # Experimentation and testing
├── data/
│   ├── raw/             # Original datasets
│   └── processed/       # Cleaned data
├── models/              # Trained models
├── templates/           # HTML frontend
├── app.py              # Flask application
└── config.py           # Configuration settings
```

## Module Documentation

### Module 1: Data Preparation and Vector Search

**Objective**: Clean e-commerce data and create vector database for semantic search

**Components**:
- `data_cleaner.py`: Cleans 541,910 rows → 401,569 clean records (73% retention)
- `vector_service.py`: Generates embeddings and manages Pinecone database
- `recommendation_service.py`: Provides natural language product recommendations

**Key Metrics**:
- Dataset: 401,569 clean transactions
- Products: 3,684 unique items
- Embedding Model: all-MiniLM-L6-v2 (384 dimensions)
- Similarity Metric: Cosine similarity
- Average accuracy: 80%+ similarity scores

### Module 2: OCR and Data Collection

**Objective**: Extract text from images and scrape training data

**Components**:
- `ocr_service.py`: Donut transformer model for text extraction
- `scraper.py`: Automated image collection for CNN training

**Capabilities**:
- Handwritten text recognition
- Multi-language support
- Automated dataset creation (50 images/category)

### Module 3: CNN Model Development

**Objective**: Train image classification model for product detection

**Components**:
- `cnn_service.py`: Complete training pipeline with validation
- `detection_service.py`: Inference service for product classification

**Model Architecture**:
- Base: MobileNetV2 (from scratch, no pre-trained weights)
- Input: 224x224 RGB images
- Classes: 8 product categories
- Validation split: 20%

**Training Features**:
- Validation metrics (accuracy, F1, precision, recall)
- Model checkpointing
- Training history logging
- Early stopping

## API Endpoints

### Endpoint 1: Text Query
```
POST /recommend
Input: {"query": "red heart decoration"}
Output: {
    "success": true,
    "products": [...],
    "response": "Natural language response"
}
```

### Endpoint 2: OCR Query
```
POST /ocr-recommend
Input: Image file with handwritten text
Output: {
    "extracted_text": "...",
    "products": [...],
    "response": "..."
}
```

### Endpoint 3: Product Detection
```
POST /detect-product
Input: Product image
Output: {
    "detected_class": "mug",
    "confidence": 0.95,
    "similar_products": [...]
}
```

## Installation and Setup

### Prerequisites
- Python 3.11+
- Conda (recommended for M2 Mac)

### Environment Setup
```bash
conda create -n ml_env python=3.11
conda activate ml_env
pip install -r requirements.txt
```

### Configuration
Create `config.py` with:
```python
PINECONE_API_KEY = "your-api-key"
INDEX_NAME = "ecommerce-products"
DIMENSION = 384
METRIC = "cosine"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Running the Application
```bash
python app.py
# Access at http://localhost:5001
```

## Usage Examples

### Data Cleaning
```python
from services.module1.data_cleaner import DataCleaner

cleaner = DataCleaner()
main_path, cnn_path = cleaner.run_cleaning_pipeline()
```

### Vector Database Setup
```python
from services.module1.vector_service import VectorService

service = VectorService()
service.create_index()
service.upload_products("data/processed/cleaned_data.csv")
```

### CNN Training
```python
from services.module3.cnn_service import train_model

results = train_model(
    images_dir="cnn_data/raw",
    num_epochs=10,
    batch_size=16
)
```

### Product Detection
```python
from services.module3.detection_service import DetectionService

detector = DetectionService()
predicted_class = detector.predict("product_image.jpg")
```

## Development Best Practices

- **Type Hints**: All functions include type annotations
- **Docstrings**: Google-style documentation for all classes/methods
- **Logging**: Comprehensive logging instead of print statements
- **Error Handling**: Try-except blocks with informative error messages
- **Modularity**: Separation of concerns across service classes
- **Configuration**: Centralized config management

## Performance Metrics

### Module 1: Vector Search
- Query time: <100ms
- Accuracy: 80%+ similarity scores
- Database: 3,684 product vectors

### Module 3: CNN Model
- Training time: ~5 minutes (10 epochs)
- Validation accuracy: Tracked per epoch
- Model size: ~14MB

## Future Improvements

1. Add more product categories
2. Implement batch prediction API
3. Add caching for frequent queries
4. Expand multilingual OCR support
5. Implement A/B testing framework

## License

This project was developed as part of a technical assessment.

## Contact

For questions or issues, please contact the development team.
