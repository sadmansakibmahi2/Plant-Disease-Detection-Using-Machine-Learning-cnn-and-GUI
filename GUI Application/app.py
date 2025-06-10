###   python -m streamlit run app.py


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import textwrap
import logging
from io import BytesIO
import base64

# Set base path to model and JSON
main_path_of_project = "C:\\Users\\user\\Desktop\\mahi's work\\python\\Data Science projects\\plant disease classification in plant village dataset with CNN\\Advanced Model\\EfficientNetB3 model\\GUI Application\\"

# Setup logging
logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# List of class labels
class_labels = [
    'Pepper bell Bacterial spot',
    'Pepper bell healthy',
    'Potato Early blight',
    'Potato Late blight',
    'Potato healthy',
    'Tomato Early blight',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite',
    'Tomato Target Spot',
    'Tomato Tomato mosaic virus',
    'Tomato healthy',
    'potato hollow heart'
]

# Paths
tf_model_path = main_path_of_project + "tf_lite_Optimize_DEFAULT_model.tflite"
json_file_path = main_path_of_project + "dis.json"

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=tf_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load cure information from JSON
@st.cache_data
def load_cure_info():
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Preprocess the image for TFLite model input
def preprocess_image(img: Image.Image, input_shape):
    img = img.resize((input_shape[1], input_shape[2]), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction with TFLite model
def classify_image_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    return class_labels[predicted_class_index], confidence

# Word wrap long cure info for better formatting
def format_cure_info(cure_info):
    return "\n".join(textwrap.wrap(cure_info, width=80))

# Display image centered in fixed size (224x224)
def display_centered_image(img: Image.Image):
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_b64}" width="224" height="224" />
            <p><em>🖼 Uploaded Image (224x224)</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit App
def main():
    st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")
    st.title("🌿 Leaf Disease Classifier")
    st.write("Upload a leaf image to detect disease and view cure suggestions.")

    interpreter = load_model()
    dis_DB_cure = load_cure_info()

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Failed to open image. Please upload a valid image file.")
            st.stop()

        display_centered_image(image)

        input_shape = interpreter.get_input_details()[0]['shape']
        img_array = preprocess_image(image, input_shape)

        with st.spinner("🔎 Classifying..."):
            try:
                predicted_class, confidence = classify_image_tflite(interpreter, img_array)
            except Exception:
                st.error("❌ Model inference failed.")
                st.stop()

        disease_name = predicted_class.lower()
        cure_info = (
            "Leaf is healthy. No cure information available."
            if "healthy" in disease_name
            else dis_DB_cure.get(disease_name, "No cure information available.")
        )

        # Log result
        logging.info(f"Prediction: {predicted_class} | Confidence: {confidence:.4f} | File: {uploaded_file.name}")

        # Display results
        st.success(f"✅ **Predicted Disease**: {predicted_class}")
        st.markdown(f"**🧠 Confidence**: `{confidence * 100:.2f}%`")
        st.markdown("### 💊 Cure Information")
        st.info(format_cure_info(cure_info))

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & TensorFlow Lite | Developed by Sadman Sakib Mahi")

if __name__ == "__main__":
    main()
