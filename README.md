# ♻️ Waste Image Classification

An AI-powered waste image classification system that uses deep learning and computer vision to automatically identify the type of waste from an uploaded image.

The project uses a **pre-trained ResNet-50 model with transfer learning** and provides an interactive **Streamlit web application** where users can upload a waste image and receive a predicted waste category.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
[Click here to try the application](https://waste-image-classification-8knjdoasyyj6cjuck4fipf.streamlit.app/)

---

## 📌 Project Overview

Waste segregation is an important step in effective recycling and waste management. Traditional waste segregation often relies on manual sorting, which can be time-consuming and prone to human error.

This project explores the use of **deep learning and computer vision** to automate waste image classification.

The trained model classifies waste images into six categories:

- 📦 Cardboard
- 🍷 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

The model is based on **ResNet-50**, a deep convolutional neural network, and uses **transfer learning** from ImageNet to improve image classification performance.

The research implementation achieved approximately **96% accuracy on the test data**.

---

## ✨ Features

- 🖼️ Upload waste images through an interactive web interface
- 🔍 Preview the uploaded image before prediction
- 🤖 AI-based waste classification
- ♻️ Classification into six waste categories
- ⚡ Fast prediction using a trained ResNet-50 model
- 🌱 Simple and user-friendly interface
- 💻 Interactive Streamlit application
- 📱 Accessible through a web browser

---

## 🧠 Machine Learning Approach

The project uses **transfer learning with ResNet-50**.

Instead of training a deep neural network completely from scratch, a ResNet-50 model pre-trained on ImageNet is used as the starting point. The final classification layer is modified to predict the six waste categories used in this project.

### Model

**Architecture:** ResNet-50  
**Framework:** PyTorch  
**Learning Approach:** Transfer Learning  
**Optimization:** Stochastic Gradient Descent (SGD)

ResNet-50 uses residual connections, which help deep neural networks learn effectively by allowing information and gradients to pass through shortcut connections.

---

## 📊 Dataset

The project uses a **Garbage Classification dataset from Kaggle** containing images belonging to six waste categories.

### Waste Categories

| Category | Description |
|----------|-------------|
| 📦 Cardboard | Cardboard waste |
| 🍷 Glass | Glass waste |
| 🥫 Metal | Metal waste |
| 📄 Paper | Paper waste |
| 🧴 Plastic | Plastic waste |
| 🗑️ Trash | General/food waste |

### Dataset Split

| Dataset | Number of Images |
|---------|------------------:|
| Training | 1,593 |
| Validation | 176 |
| Testing | 758 |
| **Total** | **2,527** |

---

## 🔄 Image Preprocessing

Before being provided to the neural network, the images undergo preprocessing to make the input consistent.

The preprocessing pipeline includes:

1. Image resizing
2. Conversion into tensor format
3. Normalization
4. Mini-batch loading
5. Batch shuffling

Images are resized to **256 × 256 pixels** before being processed by the model.

---

## 🏗️ System Workflow

The complete application works through the following steps:

```text
User uploads image
        ↓
Image Preview
        ↓
Image Preprocessing
        ↓
ResNet-50 Model
        ↓
Feature Extraction
        ↓
Classification
        ↓
Predicted Waste Category
