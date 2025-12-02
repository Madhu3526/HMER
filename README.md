# Handwritten Mathematical Expression Recognition

## Project Overview

This project focuses on converting handwritten mathematical expressions into digital LaTeX using three different machine learning approaches:

     -> Multi-Layer Perceptron (MLP) – Baseline character classifier
     -> Basic Convolutional Neural Network (CNN) – Stronger character-level OCR
     -> Encoder–Decoder with Attention (ResNet50V2 + LSTM) – End-to-end Image-to-LaTeX model

## Dataset: MathWriting (Derived from CROHME)

Total Samples: ~4,45,538 mathematical expressions

Source Format: InkML (converted to PNG)

Input: PNG images of handwritten formulas

Output: Ground-truth LaTeX sequence

Character Vocabulary (MLP/CNN): 82 classes

LaTeX Vocabulary (Seq2Seq): 64,000+ unique tokens

## Approach 1 — Multi-Layer Perceptron (MLP)

A segmentation-based pipeline for isolated character recognition.

🔹 Workflow

Binarization using adaptive thresholding

Character extraction using cv2.findContours

Cropping + resizing to 45×45

Flattening to a 2025-dim vector

MLP classification

🔹 Model

2025 → 256 → 128 → 82

ReLU activations + Dropout

🔹 Performance

✔ ~98% accuracy on isolated characters
❌ Fails for full 2D mathematical structure

## Approach 2 — Basic CNN

Improved character-level OCR using convolutional feature extraction.

🔹 Architecture

3× Conv2D(32, 3×3) + MaxPooling

Dense(128) → Dense(82)

🔹 Performance

✔ ~95% accuracy
❌ Still segmentation-based → cannot understand fractions, roots, superscripts

## Approach 3 — Encoder–Decoder with Attention (Final Model)

A complete Image-to-LaTeX deep learning system.

🔹 Encoder

ResNet50V2 pretrained on ImageNet

Extracts high-level 2D spatial features

🔹 Attention

Bahdanau Attention highlights relevant spatial regions at each decoding step

🔹 Decoder

LSTM generating LaTeX token-by-token

Vocabulary size: 64k+ tokens

🔹 Training

5 epochs (feature extraction)

3 epochs (fine-tuning)

Final Training Loss: 0.538

Validation Loss: 0.870

🔹 Evaluation

Exact Match Rate (EMR): 9.50%

Character Error Rate (CER): 80.17%
(Low EMR but demonstrates strong structural understanding; under-trained due to model complexity)
