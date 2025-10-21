import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import textwrap
import logging
from io import BytesIO
import base64
import pandas as pd

# =========================================
# 🔧 CONFIGURATION
# =========================================
main_path_of_project = "C:\\Users\\user\\Desktop\\mahi's work\\python\\Data Science projects\\plant disease classification in plant village dataset with CNN\\Advanced Model\\EfficientNetB3 model\\GUI Application\\"

# Logging
logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Labels
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

# Model & JSON Paths
tf_model_path = main_path_of_project + "tf_lite_Optimize_DEFAULT_model.tflite"
json_file_path = main_path_of_project + "dis.json"
data_info_json_path =  main_path_of_project + "datainfo.json"  # Must be in same folder as app


# =========================================
# 🧠 MODEL & DATA LOADERS
# =========================================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=tf_model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_dis_json():
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"⚠️ Could not load dis.json: {e}")
        return {}


@st.cache_data
def load_data_info():
    try:
        with open(data_info_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data["data"])
    except Exception as e:
        st.error(f"⚠️ Could not load data_info.json: {e}")
        return pd.DataFrame()


# =========================================
# 🖼 IMAGE PROCESSING & PREDICTION
# =========================================
def preprocess_image(img: Image.Image, input_shape):
    img = img.resize((input_shape[1], input_shape[2]), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    return class_labels[predicted_class_index], confidence


def format_text_block(text):
    return "\n".join(textwrap.wrap(str(text), width=80))


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


# =========================================
# 🌿 MAIN STREAMLIT APP
# =========================================
def main():
    st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")
    st.title("🌿 Leaf Disease Classifier")
    st.write("Upload a leaf image to detect disease and view cure suggestions.")

    interpreter = load_model()
    dis_DB_cure = load_dis_json()
    data_info_df = load_data_info()

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

        # Log result
        logging.info(f"Prediction: {predicted_class} | Confidence: {confidence:.4f} | File: {uploaded_file.name}")

        # ================================
        # 🌱 DISPLAY RESULTS
        # ================================
        st.success(f"✅ **Predicted Disease**: {predicted_class}")
        st.markdown(f"**🧠 Confidence**: `{confidence * 100:.2f}%`")

        # =======================================
        # 📗 1️⃣ Show details from data_info.json
        # =======================================
        st.markdown("### 📘 Disease Information (from data_info.json)")

        matched_info = data_info_df[
            data_info_df["name"].str.contains(disease_name, case=False, na=False)
        ]

        if not matched_info.empty:
            info = matched_info.iloc[0]
            st.markdown(f"**🌱 Plant:** {info['plantName']}")
            st.markdown(f"**🍃 Part Affected:** {info['plantPart']}")
            st.markdown(f"**🧫 Disease Type:** {info['diseaseType']}")
            st.markdown(f"**💊 Cure (Detailed):**")
            st.info(format_text_block(info['cure']))
        else:
            st.warning("⚠️ No matching details found in data_info.json.")

        # =======================================
        # 💊 2️⃣ Show extra cure info from dis.json
        # =======================================
        st.markdown("### 💊 Additional Cure Information (from dis.json)")
        if disease_name in dis_DB_cure:
            st.info(format_text_block(dis_DB_cure[disease_name]))
        else:
            # Try fuzzy match (contains)
            found = None
            for key in dis_DB_cure.keys():
                if disease_name in key.lower():
                    found = dis_DB_cure[key]
                    break
            if found:
                st.info(format_text_block(found))
            else:
                st.warning("⚠️ No extra cure info found in dis.json.")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & TensorFlow Lite | Developed by Sadman Sakib Mahi")


# =========================================
# 🚀 RUN APP
# =========================================
if __name__ == "__main__":
    main()
