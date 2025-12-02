
# 🌱 **Plaintify Plant Disease Classifier**

### **User Guide & Application Overview**

The Plaintify Leaf Disease Classifier is a Streamlit-based graphical interface powered by a deep-learning Keras model.
It allows farmers, researchers, agronomists, and students to quickly diagnose plant diseases simply by uploading a leaf image.

The system provides:

* 🌿 Accurate disease classification
* 📘 Rich disease information
* 💊 Cure recommendations
* 🖼 Clear visual preview
* 🧠 Confidence score
* 🧾 Auto-logging

---

# 🔧 **1. Requirements**

Before using the app, ensure you have:

* **Python 3.10 or 3.11**
* **Internet browser** (Chrome, Firefox, Edge)
* The following installed libraries.

---

# 📥 **2. Installing Python (If not installed)**

Download & install Python from:
👉 [https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation, **check the box**:

✔ *Add Python to PATH*

---

# 📦 **3. Installing Required Libraries**

Open Command Prompt or Terminal and run:

```bash
pip install streamlit tensorflow pillow numpy pandas
```

If TensorFlow fails on your system:

```bash
pip install tensorflow==2.15
```

---

# 📁 **4. Project Folder Structure**

Create a folder (example: `Plaintify-App`) and put the following files inside:

```
Plaintify-App/
│── app.py
│── Plaintify_diseases_classifier_model.keras
│── dis.json
│── datainfo.json
│── (optional) sample images/
```

Make sure the paths in `app.py` match your folder location.

---

# 🚀 **5. Running the Application**

Open a command prompt window and navigate to your project folder:

```bash
cd path/to/Plaintify-App
```

Then run the app using:

```bash
python -m streamlit run app.py
```

After a few seconds, Streamlit will launch a browser window automatically at:

👉 [http://localhost:8501](http://localhost:8501)

Your application is now running.

---

# 🖥 **6. How to Use the GUI**

### **Step 1 — Upload a Leaf Image**

Click:

➡ **“Upload an image”**

Supported formats:

* JPG
* JPEG
* PNG

The image will be displayed at the top of the interface.

---

### **Step 2 — Automatic Prediction**

Once you upload an image, the app will:

* Preprocess the image
* Run it through your Keras model
* Detect the disease
* Compute confidence percentage
* Cross-match disease names with JSON data
* Show extended details

No buttons—everything is automated.

---

### **Step 3 — Read the Output**

You will see:

#### ✔ **Predicted Disease**

Example:
**Potato Early Blight**

#### ✔ **Confidence Score**

Example:
**92.45%**

#### ✔ **Disease Information (from datainfo.json)**

* Plant name
* Part affected
* Disease type
* Long description / cure text

#### ✔ **Extra Cure Details (from dis.json)**

If exact match is missing, fuzzy matching is applied.

---

# 🧾 **7. Auto Logging**

Every prediction is recorded inside `predictions.log` with:

* Disease name
* Confidence
* File name
* Timestamp

Useful for:

* Dataset building
* Agriculture monitoring
* Student experimentation

---

# 🔄 **8. Troubleshooting Guide**

### ❌ *ModuleNotFoundError: streamlit*

Run:

```bash
pip install streamlit
```

---

### ❌ *TensorFlow Error*

Try:

```bash
pip install tensorflow==2.15
```

Or install using CPU version:

```bash
pip install tensorflow-cpu
```

---

### ❌ *Model not loading*

Check that your file exists:

```
Plaintify_diseases_classifier_model.keras
```

And verify the full path inside `app.py`.

---

### ❌ *dis.json or datainfo.json not found*

Ensure JSON paths in your code are correct and files are present.

---

### ❌ *Prediction mismatch*

Your model’s class labels must match the dataset used during training.

---

# 🌍 **9. Impact of the Plaintify Model**

This tool has meaningful real-world benefits:

## ✔ **For Farmers**

* Detect diseases early → prevent crop loss
* Quick diagnosis without expert help
* Helps small farmers with limited resources

## ✔ **For Agriculture Experts**

* Rapid field analysis
* More informed decision-making
* Ability to study disease patterns

## ✔ **For Government/NGO Projects**

* Use in village-level plant health camps
* Provide advisory support to farmers
* Improve food security

## ✔ **For Students & Researchers**

* Practical demonstration of CNN image classification
* Hands-on with AI + agriculture
* Use in projects, papers, and competitions

---

# 💡 **10. Use Cases**

### **1. Smart Agriculture**

AI-powered disease detection enables automated crop monitoring.

### **2. Mobile Diagnostics**

Can be deployed on mobile apps for farmers.

### **3. Research & Development**

Helps analyze:

* Disease spread
* Crop health variation
* Efficiency of cures and treatments

### **4. Plant Clinics**

Extension officers can use it during field visits.

### **5. Education**

Students can learn machine learning, deep learning, and plant pathology in one tool.

---

# 🎉 **Your Plaintify GUI Is Ready!**
