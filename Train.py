import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Define transformations for the training data
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset
train_dataset = datasets.ImageFolder(os.getcwd() + "\\train_data", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the pre-trained model and modify the last layer for 5 classes
model = torchvision.models.resnet18(weights="DEFAULT")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5 plant classes

# Set up optimizer, criterion, and device
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Ensure the directory exists before saving the model
os.makedirs(os.getcwd() + "\\model_weights_edir", exist_ok=True)

# Save the trained model
torch.save(model.state_dict(), os.getcwd() + "\\model_weights_edir/plant_classifier.pth")
print("Model training completed!")
