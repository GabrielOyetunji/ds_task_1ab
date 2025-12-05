import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES_DIR = os.path.join(ROOT_DIR, "cnn_data", "raw")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")


def build_model(num_classes: int):
    """
    Builds a MobileNetV2 model and replaces the classifier
    so that it matches the number of target categories.
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


def train_model(
    images_dir: str = IMAGES_DIR,
    save_path: str = MODEL_PATH,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_epochs: int = 5
):
    """
    Trains a MobileNetV2 model on images stored in a folder structure.
    Each subfolder inside images_dir represents a class.
    """

    print("Using images from:", images_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=images_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError("No images found. Ensure cnn_data/raw has class folders with images.")

    print("Classes found:", dataset.classes)
    print("Number of images:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.classes)
    model = build_model(num_classes)

    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training started.")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes
        },
        save_path
    )

    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    train_model()