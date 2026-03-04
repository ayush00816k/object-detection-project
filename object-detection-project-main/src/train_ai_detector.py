"""
Training script for AI Image Detector (binary classifier: Real vs AI-generated)

Usage:
  python src/train_ai_detector.py --data_dir data/ai_detector --epochs 10 --batch_size 32

Folder layout expected:
  data/ai_detector/train/real/
  data/ai_detector/train/ai/
  data/ai_detector/val/real/
  data/ai_detector/val/ai/

Saves checkpoint to: checkpoints/ai_detector.pth
"""
import os
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(data_dir, epochs=10, batch_size=32, lr=1e-4, save_path='checkpoints/ai_detector.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Starting training: epochs={epochs}, batch_size={batch_size}, train_samples={len(train_dataset)}, val_samples={len(val_dataset)}")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total * 100

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.2f}% - val_acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved improved model to {save_path} (val_acc {val_acc:.2f}%)")

    print(f"Training complete. Best val acc: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/ai_detector', help='data directory with train/val subdirs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', default='checkpoints/ai_detector.pth')

    args = parser.parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
