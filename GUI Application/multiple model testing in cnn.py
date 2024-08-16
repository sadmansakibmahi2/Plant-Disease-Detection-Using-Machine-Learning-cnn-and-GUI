

import os
import time
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

# Hardcoded class labels
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

# Hardcoded model directory path
MODEL_DIRECTORY = "C:\\Users\\user\\Desktop\\mahi's work\\python\\Data Science projects\\plant disease classification in plant village dataset with CNN\\Advanced Model\\EfficientNetB3 model\\GUI Application"

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(600, 600)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to classify an image using a TFLite model
def classify_image_tflite(interpreter, image_path, class_labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_array = load_and_preprocess_image(image_path, target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Measure prediction time
    start_time = time.time()
    interpreter.invoke()
    prediction_time = time.time() - start_time
    
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence, prediction_time

# Function to update the GUI with classification results
def update_gui(image_path, true_label, results):
    for widget in root.winfo_children():
        if widget != button:
            widget.destroy()

    # Display the image path
    path_label = tk.Label(root, text=f"Image Path: {image_path}", font=("Helvetica", 12), fg="green")
    path_label.pack(pady=10)

    img = Image.open(image_path)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img)
    img_label.image = img
    img_label.pack()

    # Display predictions for all models with prediction time and confidence
    for model_name, (predicted_class, confidence, prediction_time) in results.items():
        result_label = tk.Label(root, text=f"{model_name}: Predicted Class: {predicted_class}", font=("Helvetica", 14, "bold"))
        result_label.pack()

        confidence_label = tk.Label(root, text=f"Confidence: {confidence * 100:.2f}% | Time: {prediction_time:.4f} sec", font=("Helvetica", 12))
        confidence_label.pack()

# Function to classify image using all models in the directory and update GUI
def classify_and_update_all(image_path, class_labels):
    true_label = image_path.split("/")[-2]  # Extracting the true label from the image path
    results = {}

    # Iterate over all TFLite models in the directory
    for model_path in find_tflite_models(MODEL_DIRECTORY):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        predicted_class, confidence, prediction_time = classify_image_tflite(interpreter, image_path, class_labels)
        
        # Store the result for each model
        model_name = os.path.basename(model_path)
        results[model_name] = (predicted_class, confidence, prediction_time)
    
    # Update the GUI with all results
    update_gui(image_path, true_label, results)

# Function to select an image using file dialog
def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    classify_and_update_all(path, class_labels)

# Function to find all .tflite models in a directory
def find_tflite_models(directory):
    tflite_models = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tflite')]
    
    if not tflite_models:
        raise ValueError("No .tflite models found in the directory.")
    
    return tflite_models

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Leaf Disease Classifier")
    root.geometry("800x600")

    button = tk.Button(root, text="Select Image", height=3, width=20, font=("Helvetica", 12, "bold"), bg="blue", fg="white", command=select_image)
    button.pack(pady=20)

    root.mainloop()
