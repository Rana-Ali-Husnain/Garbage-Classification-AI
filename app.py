
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Garbage Classification AI",
    page_icon="♻️",
    layout="centered"
)

# Class Names
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

# Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_garbage_model.keras"
    )

model = load_model()

# App Heading
st.title("♻️ Garbage Classification AI")
st.write(
    "Upload a garbage image and let AI identify its category."
)

st.divider()

# Image Upload
uploaded_file = st.file_uploader(
    "Upload Garbage Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    if st.button("Classify Image"):

        with st.spinner("AI is analyzing your image..."):

            # Image Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)

            # Same preprocessing used during training
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = predictions[predicted_index] * 100

        # Display Result
        st.success("Prediction Completed!")

        st.subheader(
            f"Predicted Category: {predicted_class.title()}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        # Top 3 Predictions
        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(predictions)[-3:][::-1]

        for index in top_indices:
            category = CLASS_NAMES[index].title()
            score = predictions[index] * 100

            st.write(f"**{category}: {score:.2f}%**")
            st.progress(float(score / 100))

st.divider()

st.caption("AI-Based Smart Waste Classification | Powered by MobileNetV2")
