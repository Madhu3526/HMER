# 📝 Handwritten Mathematical Expression Recognition

## Project Overview
This project focuses on converting handwritten mathematical expressions into digital **LaTeX** using three progressively advanced machine learning approaches:

- **Multi-Layer Perceptron (MLP)** – Baseline character classifier  
- **Basic Convolutional Neural Network (CNN)** – Stronger character-level OCR  
- **Encoder–Decoder with Attention (ResNet50V2 + LSTM)** – End-to-end Image-to-LaTeX model  

---

## 📂 Dataset: MathWriting (Derived from CROHME)

- **Total Samples:** ~4,45,538 handwritten mathematical expressions  
- **Source Format:** InkML (converted to PNG)  
- **Input:** PNG images of handwritten formulas  
- **Output:** Ground-truth LaTeX formula  
- **Character Vocabulary (MLP/CNN):** 82 unique classes  
- **LaTeX Vocabulary (Seq2Seq):** 64,000+ unique tokens  

---

## 🔧 Approach 1 — Multi-Layer Perceptron (MLP)

A segmentation-based pipeline designed for isolated character recognition.

### 🔹 Workflow
- Binarization using *adaptive thresholding*  
- Character extraction using **cv2.findContours**  
- Cropping & resizing characters to **45 × 45**  
- Flattening to a **2025-dimensional vector**  
- Classification using MLP  

### 🔹 Model Architecture
2025 → 256 → 128 → 82
(ReLU activations + Dropout layers)


### 🔹 Performance
- ✔ Achieved **~98% accuracy** on isolated characters  
- ❌ Cannot model 2D math structure (fractions, roots, superscripts)  

---

## 🔧 Approach 2 — Basic CNN

Improved character-level OCR using convolutional feature extraction.

### 🔹 Architecture
- 3 × `Conv2D(32, 3×3)` + MaxPooling  
- Fully connected: `Dense(128) → Dense(82)`  

### 🔹 Performance
- ✔ Achieved **~95% accuracy**  
- ❌ Still segmentation-based → fails for full mathematical expressions  

---

## 🔧 Approach 3 — Encoder–Decoder with Attention (Final Model)

A complete **Image-to-LaTeX** deep learning system with end-to-end learning.

### 🔹 Encoder
- **ResNet50V2** pretrained on ImageNet  
- Extracts high-level 2D spatial features  

### 🔹 Attention
- **Bahdanau Attention**  
- Focuses on relevant image regions during token generation  

### 🔹 Decoder
- **LSTM** generating LaTeX tokens sequentially  
- Vocabulary size: **64k+ tokens**  

### 🔹 Training
- **5 epochs** – Encoder feature extraction  
- **3 epochs** – Fine-tuning  
- **Training Loss:** 0.538  
- **Validation Loss:** 0.870  

### 🔹 Evaluation
- **Exact Match Rate (EMR):** 9.50%  
- **Character Error Rate (CER):** 80.17%  
> *Despite low EMR due to dataset complexity, the model successfully learns spatial structure and generates structurally meaningful LaTeX expressions.*

---
