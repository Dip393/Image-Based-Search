import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from scipy.spatial.distance import cdist
import os

# Load the trained model for classification
@st.cache_resource
def load_classification_model():
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # 3 plant classes: Aloe Vera, Ashwagandha, Neem
    model.load_state_dict(torch.load(os.getcwd()+"\\model_weights_edir\\plant_classifier.pth"))
    model.eval()
    return model

# Initialize classification model
classification_model = load_classification_model()

# Load the feature vectors and image names for similarity search
@st.cache_data
def load_similarity_data():
    all_vecs = np.load("C:/my_project/features/all_vecs.npy")
    all_names = np.load("C:/my_project/features/all_names.npy")
    return all_vecs, all_names

vecs, names = load_similarity_data()

# Define the transformations for the uploaded image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Plant class names (adjusted to the 3 plants you have)
plant_classes = ['Aloe Vera', 'Ashwagandha', 'Neem']

# Load plant information (assuming it's stored in a text file)
def load_plant_info(plant_name):
    plant_name = plant_name.lower().replace(" ", "_")  # Converts "Aloe Vera" to "aloe_vera"
    try:
        with open(f"C:/my_project/plant_info/{plant_name}.txt", "r", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Information not available for this plant."

# Streamlit user interface
st.title("Herbal Garden Plant Recognition")

# Image upload section
uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image for classification
    img_tensor = transform(img).unsqueeze(0)

    # Perform plant classification
    with torch.no_grad():
        outputs = classification_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        plant_name = plant_classes[predicted.item()]

    st.write(f"Predicted Plant: **{plant_name}**")

    # Show plant information
    plant_info = load_plant_info(plant_name)
    st.write(plant_info)

    # Pass the image through the model to extract features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    classification_model.avgpool.register_forward_hook(get_activation("avgpool"))

    with torch.no_grad():
        classification_model(img_tensor)
        uploaded_vec = activation["avgpool"].numpy().squeeze()

    # Perform similarity search
    distances = cdist(uploaded_vec[None, ...], vecs).squeeze()
    top5_indices = distances.argsort()[:5]  # Top 5 closest images

    st.write("Top 5 Similar Images:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        image_path = "C:/my_project/images/" + names[top5_indices[i]]
        try:
            col.image(Image.open(image_path))
        except FileNotFoundError:
            col.write(f"Image not found: {names[top5_indices[i]]}")

    st.markdown(f"[More about this plant](https://tanmoy-12.github.io/herbal/{plant_name.lower().replace(' ', '')}.html)")
