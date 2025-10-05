# Plant-Disease-Detection-Using-Machine-Learning-EfficientNetB3-,GUI and web app

In this project we aim to classify plant diseases using machine learining

Its main goal is to let a user upload an image of a plant leaf, and the application will automatically identify if the leaf is healthy or has a specific disease, and then provide information on how to treat it.


### 🌿 List of Detectable Diseases

The model is trained to classify 14 different conditions across Pepper, Potato, and Tomato plants:

1.  Pepper bell Bacterial spot
2.  Pepper bell healthy
3.  Potato Early blight
4.  Potato Late blight
5.  Potato healthy
6.  Tomato Early blight
7.  Tomato Late blight
8.  Tomato Leaf Mold
9.  Tomato Septoria leaf spot
10. Tomato Spider mites Two spotted spider mite
11. Tomato Target Spot
12. Tomato Tomato mosaic virus
13. Tomato healthy
14. potato hollow heart

---

### ⚙️ Library Requirements

To run this project, you need to have the following Python libraries installed. You can install them using a `requirements.txt` file and the command `pip install -r requirements.txt`.

* **streamlit**: For creating and running the web application.
* **tensorflow**: The core machine learning framework. `tensorflow-cpu` is also a valid option if you don't have a dedicated GPU.
* **numpy**: For numerical operations, especially for handling the image array.
* **Pillow**: A fork of PIL (Python Imaging Library) used for opening, manipulating, and saving images.

### How It Works

The entire process is designed to be simple and automated:

1.  **📤 Image Upload**: You start by uploading an image (`.jpg`, `.png`, etc.) of a plant leaf using the web interface.

2.  **🧠 Model Prediction**: The application uses a powerful and efficient machine learning model called **EfficientNetB3** (optimized with TensorFlow Lite). This model analyzes the image and predicts which of the 14 possible conditions it has, such as 'Potato Late blight' or 'Tomato healthy'.

3.  **✅ Results and Cure**: The application then displays the results:
    * The **name of the disease** detected.
    * The **confidence level** of the prediction (e.g., 98.5% sure).
    * **Cure information** and treatment suggestions, which it pulls from a local JSON database file.



---

### Key Technologies Used

* **Streamlit**: A Python library used to quickly build and deploy the user-friendly web interface (the GUI).
* **TensorFlow Lite**: A lightweight version of Google's TensorFlow framework. It's used here to run the pre-trained `EfficientNetB3` model efficiently and quickly.
* **Pillow (PIL)** & **NumPy**: These libraries are used for image processing—opening the uploaded file, resizing it, and converting it into the right format for the model.
* **JSON**: A simple file format used to store the database of disease names and their corresponding cures.
