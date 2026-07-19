# 🌲 Intel Scene Classifier

An interactive web application built with **Streamlit** and **TensorFlow/Keras** that uses a Convolutional Neural Network (CNN) to classify landscape and scene images into six distinct categories.

---



## 📊 Dataset & Classes
The model is trained on the popular **Intel Image Classification** dataset. It successfully classifies images into **6 classes**:
*   🏙️ Buildings
*   🌲 Forest
*   🏔️ Glacier
*   ⛰️ Mountain
*   🌊 Sea
*   streets 🛣️ Street

---

## 🛠️ Features
*   **Instant Classification:** Upload any image (JPG, JPEG, PNG) and get immediate predictions.
*   **Confidence Metrics:** Displays the model's prediction confidence percentage.
*   **Probability Distribution:** Visual progress bars showing the model's evaluation for all 6 classes.
*   **Optimized Performance:** Uses Streamlit resource caching to build the CNN architecture and load weights smoothly without lag.

---

## 💻 Tech Stack
*   **Framework:** Streamlit
*   **Deep Learning:** TensorFlow & Keras (CNN Architecture)
*   **Image Processing:** Pillow (PIL)
*   **Data Manipulation:** NumPy

---
