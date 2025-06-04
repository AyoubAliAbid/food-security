import os
os.system("apt-get update && apt-get install -y libgl1")  # must be before cv2 import

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import Counter
import cv2
from ultralytics import YOLO
import streamlit as st

st.set_page_config(page_title="Tree Detector", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ensure this file is in the repo or download it in code

model = load_model()
class_names = model.names

# Title
st.title("🌳 Tree Detector using YOLOv8")
st.write("Upload an aerial image to detect and count trees using a trained YOLOv8 model.")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference
    results = model(image_np)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    counts = Counter(class_ids)

    # Draw results
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("fonts/Roboto-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        label = class_names[cls_id]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 3, y1 - 20), label, fill="lime", font=font)

    # Draw summary in bottom-right
    summary_text = " | ".join([f"{class_names[k]}: {v}" for k, v in counts.items()])
    text_size = draw.textbbox((0, 0), summary_text, font=font)
    padding = 10
    background_box = [
        image.width - text_size[2] - padding * 2,
        image.height - text_size[3] - padding * 2,
        image.width,
        image.height
    ]
    draw.rectangle(background_box, fill=(0, 0, 0, 200))
    draw.text(
        (background_box[0] + padding, background_box[1] + padding),
        summary_text, fill="white", font=font
    )

    # Display notifications before image
    st.success("✅ Detection complete.")
    for k, v in counts.items():
        st.info(f"{class_names[k]} count: {v}")

    # Display result image
    st.image(image, caption="Detection Result", use_column_width=True)
