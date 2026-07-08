import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# -----------------------------
# Theme Selection
# -----------------------------
theme = st.sidebar.selectbox(
    "🎨 Select Theme",
    ["Light", "Dark"]
)

# -----------------------------
# CSS
# -----------------------------
if theme == "Dark":

    background = "#0E1117"
    card = "#1E1E1E"
    text = "#FFFFFF"
    button = "#00C2FF"

else:

    background = "#F4F6FB"
    card = "#FFFFFF"
    text = "#000000"
    button = "#4F46E5"

st.markdown(
    f"""
    <style>

    .stApp{{
        background:{background};
        color:{text};
    }}

    h1,h2,h3,h4,h5,h6,p,label{{
        color:{text};
    }}

    .card{{
        background:{card};
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.2);
    }}

    .stButton>button{{
        background:{button};
        color:white;
        border-radius:10px;
        height:50px;
        width:100%;
        font-size:18px;
        border:none;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎨 Neural Style Transfer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Apply any artistic style to your photograph using AI</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Load TensorFlow Hub Model
# -----------------------------
@st.cache_resource
def load_model():

    model = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

    return model


model = load_model()

# -----------------------------
# Image Loader
# -----------------------------
def load_image(uploaded_file):

    img = Image.open(uploaded_file).convert("RGB")

    img = img.resize((512, 512))

    img = np.array(img).astype(np.float32)

    img = img / 255.0

    img = tf.convert_to_tensor(img)

    img = tf.expand_dims(img, axis=0)

    return img

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📋 Instructions")

st.sidebar.info(
    """
1. Upload Content Image

2. Upload Style Image

3. Click Generate

4. Download Result
"""
)

st.sidebar.success("TensorFlow Hub Model Loaded")

# -----------------------------
# Upload Section
# -----------------------------
left, right = st.columns(2)

with left:

    content_file = st.file_uploader(
        "📤 Upload Content Image",
        type=["jpg", "jpeg", "png"]
    )

with right:

    style_file = st.file_uploader(
        "🎭 Upload Style Image",
        type=["jpg", "jpeg", "png"]
    )

# -----------------------------
# Preview Images
# -----------------------------
if content_file is not None and style_file is not None:

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            content_file,
            caption="Content Image",
            use_container_width=True
        )

    with c2:

        st.image(
            style_file,
            caption="Style Image",
            use_container_width=True
        )

    st.success("✅ Images Uploaded Successfully")

    generate = st.button("✨ Generate Stylized Image")
    # -----------------------------
# Generate Stylized Image
# -----------------------------
if content_file is not None and style_file is not None:

    if generate:

        progress = st.progress(0)

        with st.spinner("🎨 Applying Artistic Style..."):

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            content_image = load_image(content_file)
            style_image = load_image(style_file)

            stylized_image = model(
                content_image,
                style_image
            )[0]

        progress.empty()

        output = tf.squeeze(stylized_image)
        output = tf.clip_by_value(output, 0, 1)

        output_image = Image.fromarray(
            (output.numpy() * 255).astype(np.uint8)
        )

        st.divider()

        st.subheader("🎉 Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                content_file,
                caption="📷 Content",
                use_container_width=True
            )

        with col2:
            st.image(
                style_file,
                caption="🎨 Style",
                use_container_width=True
            )

        with col3:
            st.image(
                output_image,
                caption="✨ Stylized Output",
                use_container_width=True
            )

        st.success("✅ Style Transfer Completed Successfully!")

        # -----------------------------
        # Download Button
        # -----------------------------
        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇ Download Output Image",
            data=buffer.getvalue(),
            file_name="stylized_output.png",
            mime="image/png"
        )

        st.divider()

        # -----------------------------
        # Image Details
        # -----------------------------
        st.subheader("📊 Image Information")

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Width",
                output_image.size[0]
            )

            st.metric(
                "Height",
                output_image.size[1]
            )

        with col_b:

            st.metric(
                "Format",
                "PNG"
            )

            st.metric(
                "Model",
                "TensorFlow Hub"
            )

        st.info(
            "This image was generated using Google's "
            "Arbitrary Image Stylization model."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
    """
    <div style='text-align:center;
                padding:15px;
                font-size:16px;'>

    ❤️ <b>Neural Style Transfer Web App</b><br>

    Built with <b>Streamlit</b>,
    <b>TensorFlow</b> and
    <b>TensorFlow Hub</b>

    </div>
    """,
    unsafe_allow_html=True
)