import streamlit as st
from PIL import Image
from utils import generate_caption, segment_image

st.set_page_config(page_title="Image Captioning & Segmentation", layout="centered")
st.title("🧠 Image Captioning & Segmentation App")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("📝 Generating Caption..."):
        caption = generate_caption(image.copy())
    st.success("✅ Caption Generated")
    st.markdown(f"**🗣️ Caption:** {caption}")

    with st.spinner("🔍 Segmenting Image..."):
        segmented = segment_image(image.copy())
    st.success("✅ Segmentation Done")
    st.image(segmented, caption="🖼️ Segmented Output", use_container_width=True)
else:
    st.info("Upload an image to get started.")
