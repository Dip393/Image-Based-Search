import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from scipy.spatial.distance import cdist
import os
import requests
from bs4 import BeautifulSoup
from io import BytesIO

# Load the trained model for classification
@st.cache_resource
def load_classification_model():
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)  # 5 plant classes
    model.load_state_dict(torch.load(os.getcwd() + "\\model_weights_edir\\plant_classifier.pth"))
    model.eval()
    return model


# Initialize classification model
classification_model = load_classification_model()

# Load the feature vectors and image names for similarity search
@st.cache_data
def load_similarity_data():
    all_vecs = np.load(os.getcwd() + "\\features\\all_vecs.npy")
    all_names = np.load(os.getcwd() + "\\features\\all_names.npy")
    return all_vecs, all_names


vecs, names = load_similarity_data()

# Define the transformations for the uploaded image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Plant class names for 5 plants
plant_classes = ['Aloe Vera', 'Arnica', 'Ashwagandha', 'Ginger', 'Neem']

# Load plant information (assuming it's stored in a text file)
def load_plant_info(plant_name):
    plant_name = plant_name.lower().replace(" ", "_")  # Converts "Aloe Vera" to "aloe_vera"
    try:
        with open(os.getcwd() + f"\\plant_info\\{plant_name}.txt", "r", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Information not available for this plant."


# Function to fetch Google Image URLs
def get_google_image_urls(query, num_results=5):
    """Search Google Images for a query and return a list of image URLs."""
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        st.write("Failed to fetch images from Google.")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    image_elements = soup.find_all("img", limit=num_results + 1)  # Skip the first result, which is usually a logo
    
    image_urls = []
    for img in image_elements[1:]:
        image_urls.append(img["src"])
    return image_urls


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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        plant_name = plant_classes[predicted.item()]

    st.write(f"Predicted Plant: **{plant_name}**")
    st.write(f"Prediction Confidence: **{confidence.item() * 100:.2f}%**")

    # Show plant information
    plant_info = load_plant_info(plant_name)
    st.write(plant_info)

    # Fetch and display similar images from Google
    st.write("Top 5 Similar Images:")
    top5_image_urls = get_google_image_urls(plant_name, num_results=5)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(top5_image_urls):
            try:
                # Fetch the image from the URL
                response = requests.get(top5_image_urls[i])
                img = Image.open(BytesIO(response.content))

                # Process the similar image for classification
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    outputs = classification_model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence_similar, predicted_similar = torch.max(probabilities, 1)

                col.image(img)
                col.write(f"Confidence: **{confidence_similar.item() * 100:.2f}%**")

            except Exception as e:
                col.write(f"Failed to load image {i + 1}")
        else:
            col.write(f"No image available for index {i + 1}")

    # Link to more information about the plant
    st.markdown(f"[More about this plant](https://tanmoy-12.github.io/herbal/{plant_name.lower().replace(' ', '')}.html)")

else:
    st.write("Please upload an image to identify the plant and find similar images.")
