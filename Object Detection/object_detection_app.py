






import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "Information"],
        icons=["house", "info-circle", "book"],
        menu_icon="cast",
        default_index=0,
    )

# Home page
if selected == "Home":
    st.title("Object Detector")
    st.write("Upload an image and detect objects.")

    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)
        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([], [])
        plt.yticks([], [])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.pyplot(fig, use_container_width=True)

        del prediction["boxes"]
        st.header("Predicted Probabilities")
        st.write(prediction)

# About page
elif selected == "About":
    st.title("About")
    st.write("AI object detection web app using the Python library Streamlit. I build a simple app in the tutorial that lets the user upload an image and the app detects objects present in it. The app draws bounding boxes around each object and adds labels as well. For the object detection task, I have used a pre-trained object detection model named Faster R-CNN available from PyTorch")
    st.write("The model is trained on the COCO dataset and can detect various objects such as people, cars, bikes, and more.")

# Information page
elif selected == "Information":
    st.title("Information")
    st.write("### Categories")
    st.write(", ".join(categories))
    st.write("### Model")
    st.write("The model used is Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Network (FPN).")
    st.write("### Source Code")
    st.write("You can find the source code on [GitHub](https://github.com/Saurabhraj2002?tab=repositories).")

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Created by [Saurabh Raj](https://www.linkedin.com/in/saurabh-raj-7b1894268/)
    """
)
