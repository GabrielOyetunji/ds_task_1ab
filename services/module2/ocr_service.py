"""
OCR (Optical Character Recognition) Service

Extracts text from images using Donut transformer model for handwritten query processing.
"""

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import logging
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRService:
    """Handles text extraction from images using Donut transformer model."""
    
    def __init__(self) -> None:
        """Initialize Donut OCR model and processor."""
        try:
            model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
            
            logger.info("Loading Donut model... (first time may take ~1 min)")
            
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            self.model.eval()
            self.device = "cpu"  # M2 works perfectly on CPU
            self.model.to(self.device)
            
            logger.info(f"OCRService initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCRService: {e}")
            raise

    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using Donut model.
        
        Args:
            image_path: Path to image file containing text
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If OCR processing fails
        """
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text from image
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )
            
            # Decode output
            output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            extracted_text = output.strip()
            
            logger.info(f"Extracted text: '{extracted_text}'")
            return extracted_text
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text: {e}")
    
    def process_handwritten_query(self, image_path: str) -> Optional[str]:
        """Process handwritten query image and extract search text.
        
        Args:
            image_path: Path to handwritten query image
            
        Returns:
            Extracted query text or None if extraction fails
        """
        try:
            logger.info("Processing handwritten query image")
            text = self.extract_text(image_path)
            
            if not text:
                logger.warning("No text could be extracted from image")
                return None
            
            logger.info(f"Query extracted: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Failed to process handwritten query: {e}")
            return None


if __name__ == "__main__":
    try:
        service = OCRService()
        
        # Test with sample image
        test_image = "uploaded_input.jpg"
        if os.path.exists(test_image):
            text = service.extract_text(test_image)
            print(f"Extracted text: {text}")
        else:
            logger.info("No test image found. Upload an image to test.")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
