# Import required libraries
import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests
import pandas as pd


# Streamlit Page Config
st.set_page_config(
    page_title="Computer Vision Classifier",
    page_icon="📷",
    layout="centered"
)

st.title("Real-Time Webcam Image Classification")
st.write("Using Pretrained ResNet-18 (ImageNet)")


LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
imagenet_labels = response.text.splitlines()

# Load pretrained ResNet-18

model = models.resnet18(pretrained=True)
model.eval()  

# Image preprocessing pipeline

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# STEP 5: Capture image from webcam
st.subheader("Capture Image from Webcam")
image_file = st.camera_input("Take a picture")

if image_file is not None:
    # Load and display image
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # add batch dimension

    # Model prediction
   
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get Top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Prepare results table
    results = []
    for i in range(5):
        results.append({
            "Rank": i + 1,
            "Label": imagenet_labels[top5_catid[i]],
            "Probability (%)": round(top5_prob[i].item() * 100, 2)
        })

    df = pd.DataFrame(results)

    st.subheader("Top 5 Predictions")
    st.table(df)
