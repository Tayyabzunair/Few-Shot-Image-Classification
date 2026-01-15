import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import PrototypicalNet

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Few-Shot Image Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("Few-Shot Image Classification")
st.write("Prototypical Network based image classifier")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
def load_model():
    model = PrototypicalNet().to(device)

    state_dict = torch.load("prototypical_net.pth", map_location=device)
    model.load_state_dict(state_dict)  # strict=True by default

    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# -----------------------------
# Create Dummy Prototypes
# (Same embedding dim as model output)
# -----------------------------
with torch.no_grad():
    dummy = torch.zeros(1, 1, 28, 28).to(device)
    embedding_dim = model(dummy).shape[1]

num_classes = 5
prototypes = torch.randn(num_classes, embedding_dim).to(device)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    image = image.convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image)
        distances = torch.cdist(embedding, prototypes)
        pred_class = torch.argmin(distances, dim=1).item()

    return pred_class

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten character image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=200)

    if st.button("Predict"):
        pred = predict(img)
        st.success(f"Predicted Class: {pred}")

st.markdown("---")
st.caption("Few-Shot Learning using Prototypical Networks")
