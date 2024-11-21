import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = torchvision.models.resnet18(weights="DEFAULT")
num_ftrs = model.fc.in_features
model.fc = torch.nn.Identity()  # Remove the final classification layer to get features
model.eval()

# Directory containing the images
image_dir = os.getcwd()+"\\images\\"  # Adjust this path as necessary

# List to store image features and names
feature_list = []
image_names = []

# Loop through all the images in the directory
for subdir, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png', 'webp')):
            try:
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)

                # Extract features
                with torch.no_grad():
                    features = model(img_tensor).numpy().squeeze()
                    feature_list.append(features)
                    image_names.append(os.path.relpath(img_path, image_dir))  # Save relative path
            except Exception as e:
                print(f"Error processing image {file}: {e}")

# Convert to numpy arrays
all_vecs = np.array(feature_list)
all_names = np.array(image_names)

# Save the features and image names
os.makedirs(os.getcwd()+"\\features", exist_ok=True)
np.save(os.getcwd()+"\\features\\all_vecs.npy", all_vecs)
np.save(os.getcwd()+"\\features\\all_names.npy", all_names)

print("Feature extraction completed and saved.")