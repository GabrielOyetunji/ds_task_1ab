"""
CNN Training Service for Product Image Classification

This module provides a complete CNN training pipeline with proper logging,
error handling, evaluation metrics, and model checkpointing.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES_DIR = os.path.join(ROOT_DIR, "cnn_data", "raw")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")


@dataclass
class TrainingConfig:
    """Configuration for CNN training pipeline."""
    images_dir: str = IMAGES_DIR
    save_path: str = MODEL_PATH
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 10
    val_split: float = 0.2
    image_size: Tuple[int, int] = (224, 224)
    
    def to_dict(self) -> Dict:
        return {
            'images_dir': self.images_dir,
            'save_path': self.save_path,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'val_split': self.val_split,
            'image_size': self.image_size
        }


class CNNDataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = self._create_transform()
        logger.info(f"Initialized DataLoader: {config.images_dir}")
    
    def _create_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Load and split dataset into train/validation sets."""
        try:
            if not os.path.isdir(self.config.images_dir):
                raise FileNotFoundError(f"Directory not found: {self.config.images_dir}")
            
            dataset = datasets.ImageFolder(
                root=self.config.images_dir,
                transform=self.transform
            )
            
            if len(dataset) == 0:
                raise ValueError("No images found in directory")
            
            classes = dataset.classes
            logger.info(f"Found {len(dataset)} images, {len(classes)} classes")
            
            val_size = int(len(dataset) * self.config.val_split)
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            return train_loader, val_loader, classes
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


def build_model(num_classes: int) -> nn.Module:
    """Build MobileNetV2 model with custom classifier."""
    try:
        logger.info(f"Building model for {num_classes} classes")
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise


class MetricsTracker:
    """Tracks training and validation metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute accuracy, precision, recall, F1."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            return {'accuracy': float(accuracy), 'f1': float(f1)}
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {'accuracy': 0.0, 'f1': 0.0}
    
    def update(self, train_loss: float, val_loss: float, val_metrics: Dict):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1'])
    
    def save_history(self, filepath: str):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"History saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")


class CNNTrainer:
    """Main training pipeline with validation and checkpointing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.metrics_tracker = MetricsTracker()
        self.data_loader = CNNDataLoader(config)
        logger.info(f"Trainer initialized on: {self.device}")
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer) -> float:
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            try:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        
        return running_loss / len(train_loader.dataset)
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, Dict]:
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
        
        avg_loss = running_loss / len(val_loader.dataset)
        metrics = self.metrics_tracker.compute_metrics(all_labels, all_preds)
        return avg_loss, metrics
    
    def train(self) -> Dict:
        """Execute full training pipeline."""
        try:
            logger.info("="*50)
            logger.info("STARTING TRAINING")
            logger.info("="*50)
            
            train_loader, val_loader, classes = self.data_loader.load_data()
            
            model = build_model(len(classes))
            model.to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            best_val_loss = float('inf')
            
            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                
                train_loss = self._train_epoch(model, train_loader, optimizer)
                val_loss, val_metrics = self._validate_epoch(model, val_loader)
                
                self.metrics_tracker.update(train_loss, val_loss, val_metrics)
                
                logger.info(
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'classes': classes,
                        'config': self.config.to_dict()
                    }, self.config.save_path)
                    logger.info("Best model saved")
            
            history_path = self.config.save_path.replace('.pth', '_history.json')
            self.metrics_tracker.save_history(history_path)
            
            logger.info("="*50)
            logger.info("TRAINING COMPLETE")
            logger.info(f"Model: {self.config.save_path}")
            logger.info("="*50)
            
            return {
                'success': True,
                'model_path': self.config.save_path,
                'classes': classes,
                'history': self.metrics_tracker.history
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def train_model(
    images_dir: Optional[str] = None,
    save_path: Optional[str] = None,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    num_epochs: int = 10
) -> Dict:
    """Train CNN model with specified configuration."""
    config = TrainingConfig(
        images_dir=images_dir or IMAGES_DIR,
        save_path=save_path or MODEL_PATH,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    trainer = CNNTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    results = train_model()
    print(f"\nTraining complete: {results['success']}")
    print(f"Classes: {results['classes']}")
    print(f"Model: {results['model_path']}")
