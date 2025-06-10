import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import textwrap
import logging
main_path_of_project = "C:\\Users\\user\\Desktop\\mahi's work\\python\\Data Science projects\\plant disease classification in plant village dataset with CNN\\Advanced Model\\EfficientNetB3 model\\GUI Application\\"
# Setup logging
logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Load class labels
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
tf_model_path = main_path_of_project +"tf_lite_Optimize_DEFAULT_model.tflite"
# Cache model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=tf_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Cache cure database
json_file_path = main_path_of_project +"dis.json"
@st.cache_data
def load_cure_info():
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Image preprocessing
def preprocess_image(img: Image.Image, input_shape):
    img = img.resize((input_shape[1], input_shape[2]), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
def classify_image_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    return class_labels[predicted_class_index], confidence

# Word wrapping
def format_cure_info(cure_info):
    return "\n".join(textwrap.wrap(cure_info, width=80))

# Main app
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
        except Exception as e:
            st.error("❌ Failed to open image. Please upload a valid image file.")
            st.stop()

        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        input_shape = interpreter.get_input_details()[0]['shape']
        img_array = preprocess_image(image, input_shape)

        with st.spinner("🔎 Classifying..."):
            try:
                predicted_class, confidence = classify_image_tflite(interpreter, img_array)
            except Exception as e:
                st.error("❌ Failed to run model inference.")
                st.stop()

        disease_name = predicted_class.lower()
        if "healthy" in disease_name:
            cure_info = "Leaf is healthy. No cure information available."
        else:
            cure_info = dis_DB_cure.get(disease_name, "No cure information available.")

        # Logging
        logging.info(f"Prediction: {predicted_class} | Confidence: {confidence:.4f} | File: {uploaded_file.name}")

        # Output
        st.success(f"✅ **Predicted Disease**: {predicted_class}")
        st.markdown(f"**🧠 Confidence**: `{confidence * 100:.2f}%`")
        st.markdown("### 💊 Cure Information")
        st.info(format_cure_info(cure_info))

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & TensorFlow Lite  Developed by Sadman Sakib Mahi")

if __name__ == "__main__":
    main()
