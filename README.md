# 🌿 Crop Disease Detection Using CNN & Transfer Learning

## 📋 Project Overview
An AI-powered deep learning system that detects plant diseases from leaf images using Convolutional Neural Networks and Transfer Learning with MobileNetV2. Built as part of the Infotact Solutions Technical Internship Program in Bengaluru.

## 🎯 Problem Statement
Traditional crop disease detection relies on manual inspection by agricultural experts which is slow, expensive and often too late. This system enables farmers to instantly diagnose plant diseases by simply uploading a photo of a leaf through a web application, without requiring any expert knowledge.

## 📁 Dataset
- **Source:** PlantVillage Dataset (Kaggle)
- **Total Images:** 20,639
- **Total Classes:** 15 disease categories
- **Crops Covered:** Tomato, Potato, Pepper
- **Split:** 70% Train (14,447) | 15% Validation (3,096) | 15% Test (3,096)

## 📅 Week 1 — Data Collection, EDA and Preprocessing

**What was done:**
- Downloaded and extracted PlantVillage dataset containing 20,639 labeled leaf images across 15 disease classes
- Performed Exploratory Data Analysis
- Visualized sample images from each of the 15 disease classes
- Plotted class distribution bar chart and identified dataset imbalance
- Largest class had 3,209 images, smallest had approximately 150 images

**Preprocessing Pipeline:**
- Collected all image filepaths and labels into a structured Pandas DataFrame
- Applied stratified train/validation/test split to maintain class proportions
- Training: 14,447 images | Validation: 3,096 images | Testing: 3,096 images
- Configured ImageDataGenerator for training with augmentation:
  - Rotation range: 20 degrees
  - Zoom range: 0.2
  - Horizontal flip: True
  - Width and height shift: 0.2
  - Normalization: rescale 1/255
- Validation and test generators with normalization only (no augmentation)
- Batch size: 32 | Standard image size: 224x224 pixels
- Used streaming/lazy loading to avoid RAM overflow on Google Colab

## 📅 Week 2 — Custom CNN Architecture and Baseline Training

**What was done:**
- Applied Label Encoding to convert 15 disease names into numbers
- Applied One Hot Encoding to convert numbers into 15-dimensional vectors
- Built custom CNN from scratch using TensorFlow Keras Sequential model

**CNN Architecture:**
- Conv2D (32 filters, 3x3, ReLU) + MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3, ReLU) + MaxPooling2D (2x2)
- Conv2D (128 filters, 3x3, ReLU) + MaxPooling2D (2x2)
- Flatten
- Dense (128 neurons, ReLU)
- Dropout (0.5)
- Dense (15 neurons, Softmax)
- Total Parameters: 11,170,895

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Categorical Cross Entropy
- Epochs: 15
- Early Stopping: patience=3, monitor=val_loss, restore_best_weights=True

**Results:**
- Best Validation Accuracy: 87.69% at Epoch 11
- Best Validation Loss: 0.3771
- Final Test Accuracy: 86.43%
- Final Test Loss: 0.3834

## 📅 Week 3 — Transfer Learning with MobileNetV2

**What was done:**
- Implemented Transfer Learning using MobileNetV2 pre-trained on ImageNet
- Loaded MobileNetV2 with include_top=False to remove original classification head
- Froze all base MobileNetV2 layers to preserve features learned from 1.2 million images

**Transfer Learning Model Architecture:**
- MobileNetV2 base (frozen, pre-trained on ImageNet)
- GlobalAveragePooling2D
- Dense (256 neurons, ReLU)
- Dropout (0.5)
- Dense (15 neurons, Softmax)

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Categorical Cross Entropy
- Epochs: 10
- Early Stopping: patience=3, monitor=val_loss, restore_best_weights=True

**Results:**
- Significantly higher accuracy than Week 2 baseline CNN
- Model saved as crop_disease_mobilenet_model.h5
- Class names saved as class_names.npy for deployment

## 📅 Week 4 — Streamlit Web Application and Deployment

**What was done:**
- Built interactive web application using Streamlit
- Deployed publicly using ngrok tunneling from Google Colab

**App Features:**
- Image upload interface accepting JPG, JPEG, PNG formats
- Displays uploaded image to the user
- Preprocessing pipeline inside app:
  - Convert to RGB
  - Resize to 224x224
  - Normalize pixels from 0-255 to 0-1
  - Add batch dimension using np.expand_dims
- Loads saved MobileNetV2 model and class names
- Runs prediction and finds highest probability class using np.argmax
- Displays predicted disease name in green success box
- Displays confidence score in blue info box

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, PIL |
| Data Manipulation | NumPy, Pandas |
| Visualization | Matplotlib |
| ML Utilities | Scikit-learn |
| Web App | Streamlit |
| Deployment | Ngrok |
| Environment | Google Colab |


## 📦 Disease Classes Detected

1. Pepper Bell Bacterial Spot
2. Pepper Bell Healthy
3. Potato Early Blight
4. Potato Late Blight
5. Potato Healthy
6. Tomato Bacterial Spot
7. Tomato Early Blight
8. Tomato Late Blight
9. Tomato Leaf Mold
10. Tomato Septoria Leaf Spot
11. Tomato Spider Mites Two Spotted
12. Tomato Target Spot
13. Tomato Mosaic Virus
14. Tomato Yellow Leaf Curl Virus
15. Tomato Healthy

## ⚙️ Key Technical Decisions

| Decision | Reason |
|----------|--------|
| Stratified Split | Dataset was imbalanced so each split needed proportional class representation |
| Augmentation on training only | Validation and test sets must reflect real unmodified images |
| ImageDataGenerator streaming | Avoids loading 20,000 plus images into RAM simultaneously |
| Image size 224x224 | Standard input size required by MobileNetV2 |
| Dropout 0.5 | Prevents overfitting by randomly disabling 50% of neurons during training |
| Early Stopping patience 3 | Stops wasted training epochs and restores best weights automatically |
| Frozen MobileNetV2 layers | Preserves ImageNet knowledge and prevents catastrophic forgetting |
| GlobalAveragePooling2D | More compact than Flatten and reduces overfitting in transfer learning |
| fit only on training data | Prevents data leakage in Label Encoding |

## ⚠️ Challenges Faced and Solutions

| Challenge | Solution |
|-----------|----------|
| Dataset imbalance — some classes had 20x more images than others | Used stratified splitting and data augmentation to handle imbalance |
| RAM overflow when loading 20,000 images | Switched to ImageDataGenerator streaming — loads 32 images at a time |
| BGR vs RGB color mismatch in OpenCV | Used cv2.cvtColor to convert BGR to RGB before display |
| Overfitting in custom CNN | Added Dropout(0.5) and Early Stopping with patience=3 |
| Invalid image filename warning in validation generator | Identified and acknowledged the corrupted file — it was automatically skipped |
| Streamlit not starting before ngrok connects | Added time.sleep(4) between starting Streamlit and connecting ngrok |


## 👩‍💻 Intern Details
- Name: Hazel Mehra
- Organization: Infotact Solutions and Co.
- Location: Bengaluru, Karnataka
- Program: Technical Internship — Data Science and Machine Learning
- Project: Project 1 — Computer Vision for Crop Disease Detection
