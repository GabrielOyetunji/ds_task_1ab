"""
Product Detection Service using CNN

Loads trained CNN model and performs product classification from images.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import logging
from typing import Tuple, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionService:
    """Handles product detection and classification using trained CNN model."""
    
    def __init__(self, model_path: str = None) -> None:
        """Initialize detection service with trained model.
        
        Args:
            model_path: Path to trained model file. If None, uses default path.
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            # Resolve model path
            if model_path is None:
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                model_path = os.path.join(root_dir, "models", "cnn_model.pth")
            
            logger.info(f"Loading model from: {model_path}")
            
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Load class names
            self.classes = checkpoint["classes"]
            num_classes = len(self.classes)
            logger.info(f"Model loaded with {num_classes} classes: {self.classes}")
            
            # Build MobileNetV2 model
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
            
            # Load model weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            # Same preprocessing used during training
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
            logger.info("DetectionService initialized successfully")
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DetectionService: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def predict(self, image_path: str) -> str:
        """Predict product class from image.
        
        Args:
            image_path: Path to product image
            
        Returns:
            Predicted product class name
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If prediction fails
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Predicting class for: {image_path}")
            
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img)
                pred_idx = torch.argmax(output, dim=1).item()
            
            predicted_class = self.classes[pred_idx]
            logger.info(f"Predicted class: {predicted_class}")
            
            return predicted_class
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Failed to predict: {e}")
    
    def predict_with_confidence(self, image_path: str) -> Tuple[str, float]:
        """Predict product class with confidence score.
        
        Args:
            image_path: Path to product image
            
        Returns:
            Tuple of (predicted_class, confidence_score)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If prediction fails
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Predicting with confidence for: {image_path}")
            
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img)
                probabilities = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probabilities, dim=1)
                
                pred_idx = pred_idx.item()
                confidence = confidence.item()
            
            predicted_class = self.classes[pred_idx]
            logger.info(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")
            
            return predicted_class, confidence
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Failed to predict: {e}")
    
    def get_all_classes(self) -> List[str]:
        """Get list of all product classes the model can detect.
        
        Returns:
            List of class names
        """
        return self.classes


if __name__ == "__main__":
    try:
        service = DetectionService()
        logger.info(f"Model ready with classes: {service.get_all_classes()}")
        
        # Test prediction if test image exists
        test_image = "uploaded_product.jpg"
        if os.path.exists(test_image):
            predicted_class = service.predict(test_image)
            print(f"Predicted class: {predicted_class}")
        else:
            logger.info("No test image found. Upload a product image to test.")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
