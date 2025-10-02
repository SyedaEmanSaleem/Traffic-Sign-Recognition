# 🚦 Traffic Sign Recognition

## **Overview**

Traffic Sign Recognition using **Convolutional Neural Networks (CNN)** on the **GTSRB dataset**.
This project classifies traffic signs in images, evaluates model performance, and visualizes predictions with images and video.

---

## **Features**

* Preprocessing: resizing, normalization, one-hot encoding
* CNN model with 3 convolutional blocks, Batch Normalization, and Dropout
* Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
* Evaluation: accuracy, loss, confusion matrix, classification report
* Visualization: training curves, random/misclassified test images
* Video demonstration of predictions
* Class ID mapping to human-readable traffic sign names

---

## **Dataset**

* **GTSRB – German Traffic Sign Recognition Benchmark**
* 43 traffic sign classes
* Train and Test folders with CSV files for labels

---

## **Installation**

```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
pip install -r requirements.txt
```
---

## **Usage**

1. Open the Colab notebook.
2. Run cells sequentially:

   * Import libraries
   * Set dataset paths
   * Load and preprocess data
   * Build and compile CNN model
   * Train with callbacks
   * Evaluate with accuracy, loss, and confusion matrix
   * Visualize predictions and generate a video

---

## **Results**

* Training & validation accuracy/loss plots
* Confusion matrix for evaluation of all 43 classes
* Random and misclassified test image visualizations
* Video animation of model predictions

---

## **Model Architecture**

* **Input:** 30x30x3 images
* **Conv layers:** 32 → 64 → 128 filters + ReLU + MaxPooling
* **Dense layers:** 256 neurons + Dropout
* **Output layer:** 43 classes, Softmax activation

---

## **Future Improvements**

* Use **transfer learning** (ResNet, MobileNet) for better accuracy
* Real-time traffic sign recognition via **webcam or video stream**
* Deploy as a **web app** with Streamlit/Gradio or **mobile app** using TensorFlow Lite
* Apply **Grad-CAM** to visualize model focus areas

---

## **License**

MIT License

---
