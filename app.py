
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Garbage Classification AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# PROJECT LINKS
# Apne actual URLs yahan paste karna
# ----------------------------------

LINKS = {
    "Dataset & Code Details": "https://www.kaggle.com/datasets/mostafaabla/garbage-classification",
    "Project Report": "https://docs.google.com/document/d/1PheF0D7cxeZ0KxZZm-b8pK71QTxn9fHi/edit?usp=sharing&ouid=109821129166928634241&rtpof=true&sd=true",
    "PPT / Presentation": "https://docs.google.com/presentation/d/1XeztzZH0t3pdkHCZyPKWFTUu8EkztkpO/edit?usp=sharing&ouid=109821129166928634241&rtpof=true&sd=true",
    "YouTube Video": "https://youtu.be/cmGJfcBQ1Bo",
    "GitHub Repository": "https://github.com/Rana-Ali-Husnain/Garbage-Classification-AI",
    "LinkedIn": "https://lnkd.in/p/dGKv6neB",
    "Kaggle": "https://www.kaggle.com/code/ranaalihusnain/garbage-classification/notebook"
}

# ----------------------------------
# CLASS NAMES
# ----------------------------------

CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

# ----------------------------------
# CUSTOM CSS - LIGHT THEME
# ----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
    color: #1f2937;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

.hero {
    background: linear-gradient(120deg, #e5f7eb, #f0fdf4);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #d1ead9;
    margin-bottom: 25px;
}

.hero h1 {
    color: #166534;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #4b6352;
    font-size: 16px;
}

.section-title {
    color: #166534;
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 13px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

div.stButton > button {
    background-color: #15803d;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 25px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #166534;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_garbage_model.keras"
    )

try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi ho saka. Model file ka path check karein.")
    st.exception(e)
    st.stop()

# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:
    st.title("♻️ Garbage AI")
    st.caption("Smart Waste Classification System")

    st.divider()

    st.subheader("Project Resources")

    for title, url in LINKS.items():
        if url.startswith("https://") or url.startswith("http://"):
            st.markdown(f"🔗 [{title}]({url})")
        else:
            st.caption(f"Add URL: {title}")

    st.divider()

    st.markdown("**Model Information**")
    st.write("Architecture: MobileNetV2")
    st.write("Framework: TensorFlow / Keras")
    st.write("Input Size: 224 × 224")
    st.write("Categories: 12")

    st.divider()
    st.caption("AI-Based Smart Waste Classification")

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown("""
<div class="hero">
    <h1>♻️ Garbage Classification AI</h1>
    <p>
        An intelligent waste classification system powered by
        Deep Learning and MobileNetV2.
        Upload an image to identify its waste category.
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------
# IMAGE CLASSIFICATION
# ----------------------------------

st.markdown(
    '<div class="section-title">Image Classification</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a garbage image and our AI model will automatically "
    "identify its category."
)

uploaded_file = st.file_uploader(
    "Upload Garbage Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)

        with st.spinner("AI is analyzing your image..."):

            resized_image = image.resize((224, 224))

            img_array = np.array(
                resized_image,
                dtype=np.float32
            )

            img_array = (
                tf.keras.applications.mobilenet_v2
                .preprocess_input(img_array)
            )

            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            st.write("DEBUG: Model prediction completed")
            st.write("Prediction shape:", predictions.shape)
            st.write("Prediction values:", predictions)

            st.write("DEBUG: Predicted index:", int(np.argmax(predictions)))

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        with right:
            st.subheader("AI Classification Result")

            st.success("Classification Completed!")

            st.markdown(
                f"### {predicted_class.replace('-', ' ').title()}"
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            st.markdown("**Top 3 Predictions**")

            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:

                category = CLASS_NAMES[index].replace(
                    "-", " "
                ).title()

                score = float(predictions[index]) * 100

                st.write(f"**{category}: {score:.2f}%**")

                st.progress(
                    min(max(score / 100, 0.0), 1.0)
                )

    except Exception as e:
        st.error("Image classification failed.")
        st.exception(e)

# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <h4 style="color:#166534;">
            ♻️ Garbage Classification AI
        </h4>
        <p style="color:#64748b;">
            Powered by MobileNetV2 | TensorFlow | Streamlit
        </p>
        <p style="color:#64748b; font-size:13px;">
            AI-Based Smart Waste Classification Project
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
