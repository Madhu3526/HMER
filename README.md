# ✍️➡️🧮 Handwritten Mathematical Expression Recognition

This project explores different deep learning approaches to translate images of handwritten mathematical expressions into digital formats.  
It implements two distinct methodologies:  
- a **baseline character-level classification system**, and  
- an **advanced end-to-end image-to-sequence model with attention**.

The goal is to provide a robust solution for digitizing handwritten formulas, enhancing their searchability, accessibility, and archival.

---

## 🌟 Features

### 🧩 Two Distinct Approaches Implemented

#### **Approach 1 (Baseline - Character Classification)**
- Utilizes classic computer vision (OpenCV) for contour detection and character segmentation.  
- Employs a simple Convolutional Neural Network (CNN) for individual character classification.  
- Outputs a linear string of predicted characters.

#### **Approach 2 (Advanced - Image-to-LaTeX End-to-End)**
- Translates entire handwritten formula images into LaTeX code directly.  
- Leverages an Encoder-Decoder architecture with **Bahdanau Attention**.  
- **Transfer Learning:** Uses pre-trained **ResNet50V2** for visual feature extraction.  
- **Two-Phase Training:** Combines frozen encoder training with fine-tuning.  
- Handles 2D structural complexities (fractions, exponents, integrals, etc.).  
- Includes evaluation metrics: **Character Error Rate**, **Exact Match Rate**, **BLEU Score**.  
- Interactive **Streamlit Web App** for:
  - Uploading handwritten images  
  - Predicting LaTeX formulas  
  - Dynamically entering variable values  
  - Computing results using **SymPy**

---

## 🧠 Model Architecture (Approach 2)

### **Encoder (ResNet50V2)**
- Pre-trained on ImageNet for powerful feature extraction.  
- Converts input (224×224×3) into a rich feature map (7×7×2048).  
- Early layers frozen initially, then fine-tuned later.

### **Attention Mechanism (Bahdanau)**
- Allows the decoder to focus on the most relevant parts of the image dynamically.  
- Essential for parsing spatially complex mathematical layouts.

### **Decoder (LSTM)**
- Generates LaTeX tokens one at a time (autoregressive generation).  
- Uses embeddings of previous tokens and attention context vectors.

---

## 🧰 Technology Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Backend | Python 3.9+ |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Web Interface | Streamlit |
| Math Parsing | SymPy |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Pillow |
| Metrics | Scikit-learn, NLTK (BLEU) |

