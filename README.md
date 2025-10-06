# ✍️➡️🧮 Handwritten Math Recognition

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



## 🗂️ Project Structure
.
├── app.py # Streamlit web app (Approach 2)
├── train_approach2.py # Training script for advanced model
├── train_approach1.py # Training script for baseline model
├── predict_approach1.py # Prediction script (Approach 1)
├── eda.ipynb # Exploratory Data Analysis
├── vocab.txt # Vocabulary for Approach 2
├── requirements.txt # Python dependencies
├── training_checkpoints/ # Saved model weights (Approach 2)
├── math_model/ # Saved model weights (Approach 1)
├── extracted_images/ # Segmented characters (Approach 1)
├── formula_images/ # Full formula dataset (Approach 2)
├── image/ # Sample input images
└── README.md


---

## ⚙️ Setup & Installation

### 1. Clone the Repository

git clone <your-repository-url>
cd <repository-name>

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt


💡 If you don’t have requirements.txt, create one:

pip freeze > requirements.txt

4. Download Data & Models

Approach 1 Data: extracted_images/ → segmented character images

Approach 2 Data: formula_images/ → CROHME or custom dataset

Model Checkpoints: Place trained weights and vocab.txt in root directory.

🚀 Usage
1. Running Approach 1 (Baseline)

Predict characters from an image:

python predict_approach1.py --image_path "image/image1.jpg"

2. Running Approach 2 (Advanced)
a) Using Streamlit Web App (Recommended)
streamlit run app.py


Then open your browser → upload handwritten formula → get LaTeX → calculate result.

b) Using Command-Line Prediction
python train_approach2.py --mode predict --image_path "image/formula.png"


Or use a dedicated predict_approach2.py script if available.

3. Training (Optional)

Train baseline:

python train_approach1.py


Train advanced model:

python train_approach2.py

📊 Results & Performance
Approach 1 (Baseline)

✅ Output Example:
Input: y² + Ny₁ + 1 = N
Output: y2+Ny1+1=N

⚠️ Limitation:
Struggles with 2D math structures; only linear character predictions.

Approach 2 (Advanced - Image-to-LaTeX)
Metric	Score
Exact Match Rate	9.50%
Character Error Rate	80.17%
BLEU Score	0.0142

✅ Output Example: \frac{d}{dt}N_{2}

📈 Analysis:
Demonstrates learning of LaTeX grammar and 2D layouts via attention.
Further training expected to reduce CER and improve accuracy.

🔮 Future Work

Train Approach 2 longer on larger dataset for better convergence.

Apply data augmentation (stroke thickness, noise, deformation).

Try Transformer-based decoders for long-range dependency modeling.

Implement beam search for improved predictions.

Strengthen baseline segmentation and CNN pipeline.

Develop an error analysis visualization tool.
