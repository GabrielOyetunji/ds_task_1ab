import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class DetectionService:

    def __init__(self):
        # Resolve model path
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(root_dir, "models", "cnn_model.pth")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        # Load class names
        self.classes = checkpoint["classes"]
        num_classes = len(self.classes)

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

    def predict(self, image_path: str) -> str:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            pred_idx = torch.argmax(output, dim=1).item()

        return self.classes[pred_idx]