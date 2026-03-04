
# 🌱 Plant-Disease-Detection-Using-Machine-Learning-CNN and Web App

In this project, we aim to accurately **classify plant diseases using Machine Learning and Convolutional Neural Networks (CNNs)**.
The system uses a trained deep learning model (`plantify_model.keras`) to recognize multiple plant leaf diseases from an uploaded image.

##  App Demo



https://github.com/user-attachments/assets/7702983c-4c49-42e8-a6a1-edde2e64505a




Model Development Code: https://www.kaggle.com/code/sadmansakibmahi/acc-99-36-novel-cnn-model-for-plant-disease-clf

Dataset: https://www.kaggle.com/datasets/sadmansakibmahi/plant-disease-expert

Deployed Model: https://www.kaggle.com/models/sadmansakibmahi/plaintify-diseases-classifier-model



The main goal of this project is simple but powerful:
➡️ **Allow a user to upload an image of a plant leaf and automatically detect whether the leaf is healthy or infected with a specific disease, and optionally provide treatment suggestions.**

This tool can assist farmers, students, researchers, and agricultural workers in making faster and more informed decisions—preventing crop losses and improving yield quality.

---

# 🌿 **List of Detectable Diseases**

The model is trained to classify **59 different plant conditions** across a wide variety of crops, including Apple, Rice, Corn, Grape, Tomato, Pepper, Potato, Tea, Orange, Strawberry, Soybean, and more.

Below is the full list of detectable plant conditions as defined in your class labels:

1. Apple Apple scab
2. Apple Black rot
3. Apple Cedar apple rust
4. Apple healthy
5. Bacterial leaf blight in rice leaf
6. Blight in corn Leaf
7. Blueberry healthy
8. Brown spot in rice leaf
9. Cercospora leaf spot
10. Cherry Powdery mildew
11. Cherry healthy
12. Common Rust in corn Leaf
13. Corn (maize) healthy
14. Garlic
15. Grape Black rot
16. Grape Esca Black Measles
17. Grape Leaf blight (Isariopsis)
18. Grape healthy
19. Gray Leaf Spot in corn leaf
20. Leaf smut in rice leaf
21. Nitrogen deficiency in plant
22. Orange Haunglongbing (Citrus greening)
23. Peach healthy
24. Pepper bell Bacterial spot
25. Pepper bell healthy
26. Potato Early blight
27. Potato Late blight
28. Potato healthy
29. Raspberry healthy
30. Sogatella rice
31. Soybean healthy
32. Strawberry Leaf scorch
33. Strawberry healthy
34. Tomato Bacterial spot
35. Tomato Early blight
36. Tomato Late blight
37. Tomato Leaf Mold
38. Tomato Septoria leaf spot
39. Tomato Spider mites (Two-spotted spider mite)
40. Tomato Target Spot
41. Tomato Mosaic virus
42. Tomato healthy
43. Waterlogging in plant
44. Algal leaf in tea
45. Anthracnose in tea
46. Bird eye spot in tea
47. Brown blight in tea
48. Cabbage looper
49. Corn crop
50. Ginger
51. Healthy tea leaf
52. Lemon canker
53. Onion
54. Potassium deficiency in plant
55. Potato crop
56. Potato hollow heart
57. Red leaf spot in tea
58. Tomato canker
59. Healthy plant (general class if included by your model)

---


### ⚙️ Library Requirements

To run this project, ensure you have the following Python libraries installed.
You can install everything using:

```
pip install -r requirements.txt
```

**Required Libraries**

* **streamlit** – For building and running the interactive web app
* **tensorflow / keras** – For loading and running the plant disease classification model
* **numpy** – For numerical operations and image array handling
* **Pillow (PIL)** – For opening and resizing uploaded images

---

### ⚙️ How It Works

The workflow of the project is optimized to be smooth, simple, and intuitive:

#### 1. 📤 Image Upload

The user uploads a plant leaf image (`.jpg`, `.jpeg`, `.png`, etc.) through the Streamlit interface.

#### 2. 🧠 Model Prediction

The backend uses a **CNN-based Keras model** (`plantify_model.keras`) to analyze the leaf image.
The image is preprocessed, resized to `150x150`, normalized, and passed through the neural network.

The model then predicts which of the 14 disease categories the leaf belongs to.

#### 3. 📊 Results Display

The app shows:

* The **predicted disease name**
* The **confidence percentage**
* The **uploaded leaf image** for confirmation

#### 4. 💡 Cure or Guidance (Optional Feature)

If you connect a JSON treatment database, the system can also provide:

* Cure information
* Prevention steps
* Additional notes

---

### 🛠️ Key Technologies Used

* **Streamlit** – For building a clean, fast, and interactive graphical web app
* **TensorFlow / Keras** – For loading and running the trained CNN model
* **NumPy** – For array manipulation and model input preparation
* **Pillow (PIL)** – For reading and resizing the images
* **JSON (Optional)** – For linking disease names to cure information

---

### ▶️ How to Run the Web App

1. Install the required libraries:

```
pip install -r requirements.txt
```

2. Place the following files in the project directory:

* `app.py` (your Streamlit app)
* `plantify_model.keras` (the trained model)
* `class_names.py` or JSON file containing class label names

3. Run the Streamlit server:

```
python -m streamlit run app.py
```

4. Your browser will automatically open the web interface.

---

### 🌍 Impact & Use Cases

Plant diseases cause billions of dollars in crop losses each year.
This system provides an efficient, low-cost, AI-powered solution with real-world benefits:

#### 🌾 For Farmers

Early detection minimizes crop loss and improves productivity.

#### 🎓 For Students & Researchers

Useful for agriculture research, ML experiments, and academic projects.

#### 🧪 For Developers

A practical example of CNN image classification integrated into a real web app.

#### 🌍 For Communities

Can be deployed in low-resource regions to support local growers and agricultural workers.

---


