import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model

# setting PathS
# Use full paths based on current project root
train_dir = os.path.join(os.getcwd(), "data", "images", "train")
test_dir = os.path.join(os.getcwd(), "data", "images", "test")
# DATA TRANSFORMS

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),  # Changes every image to 128×128 pixels.
    transforms.RandomHorizontalFlip(),  # Rotate randomly in-between images
    transforms.RandomRotation(20), # Rotates the image randomly up to ±20 degrees.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randomly changes saturation,brightness,contrast
    transforms.ToTensor(),  # Converts image from PIL format → PyTorch tensor.(numbers)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # It shifts numbers from 0–1 → -1 to 1 (HELPS IN BALANCING THE NUMBERS)
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)), # resizing images to 128x128 size
    transforms.ToTensor(), #converts to torch numbers 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  #same as in train
])

# LOAD DATA
train_data = datasets.ImageFolder(train_dir, transform=transform_train) # reads all images from your train folder
test_data = datasets.ImageFolder(test_dir, transform=transform_test) # reads all images from your test folder
train_loader = DataLoader(train_data, batch_size=4, shuffle=True) # Loads images in small batches of 4. and randomly mixes images every epoch(shuffling)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False) # Loads images in small batches of 4. and with no shuffling

print(f"✅ Loaded {len(train_data)} training images")
print(f"✅ Loaded {len(test_data)} testing images")
print(f"📚 Classes: {train_data.classes}") # shows class names which is working on 

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # it will use the gpu if available otherwise cpu 

# TRAIN FUNCTION
def train_model(model, model_name, epochs=8,lr=0.001): # model + model name + epoch + learning rate (ability of model to learn (deep or easy))
    criterion = nn.CrossEntropyLoss() # It tells the model how wrong its predictions are.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) #The adam optimizer updates the model’s weights to reduce the loss.
    model.to(device) # move to gpu or cpu depending on availability

  # Actual Training stars from here : 
    print(f"\n🚀 Training {model_name}...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0   # total loss in this epoch

        for images, labels in train_loader: # Loop through training batches
            images, labels = images.to(device), labels.to(device) # Move the images + labels to GPU
            optimizer.zero_grad() # clear previous gradients

            outputs = model(images) # model predicts the image
            loss = criterion(outputs, labels) # compares the prediction
            loss.backward()
            optimizer.step() # learns form past mistakes

            running_loss += loss.item() # update loss and accuracy for each epoch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total # here the epoch accuracy will be shown on the screen
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss:.2f} | Accuracy: {acc:.2f}%")

    # save the trained model to checkpoints 
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"{model_name.lower()}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ {model_name} saved to {save_path}\n")


train_model(CNNModel(num_classes=len(train_data.classes)), "CNN", epochs=6, lr=0.001)  

train_model(get_resnet18_model(num_classes=len(train_data.classes)), "ResNet18", epochs=3, lr=0.001 ) 

train_model(get_mobilenet_model(num_classes=len(train_data.classes)),"MobileNet", epochs=3, lr=0.001)  

# epoch = how many times the model sees all training images.
# learning rate = how fast the model learns.