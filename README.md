# Dog vs Cat Image Classification

A deep learning project for binary image classification of **cats and dogs** using **TensorFlow and Keras**.

This project compares three different approaches:

1. Custom Convolutional Neural Network (CNN)
2. EfficientNetB0 Transfer Learning / Feature Extraction
3. ResNet50 Full Fine-Tuning

The models are evaluated using accuracy, precision, recall, and F1-score.

---

## 📌 Project Overview

The objective of this project is to build and compare different deep learning models for classifying images into two categories:

-  Cats
-  Dogs

The project covers the complete image classification workflow, including:

- Dataset preparation
- Image resizing and normalization
- Data augmentation
- Custom CNN development
- Transfer learning
- Feature extraction
- Full fine-tuning
- Model evaluation
- Model comparison
- Error analysis

---

## 📂 Dataset

The Dog vs Cat dataset is downloaded directly in the notebook from:

`https://github.com/laxmimerit/dog-cat-full-dataset`

The dataset contains two classes:

- `cats`
- `dogs`

### Image Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Number of Classes | 2 |
| Classification Type | Binary |
| Random Seed | 42 |

The dataset is divided into:

- **80% Training**
- **20% Validation**
- Independent **Test Dataset**

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- EfficientNetB0
- ResNet50

---

# 🔬 Methodology

## 1. Dataset Preparation

Images are loaded using:

`tf.keras.utils.image_dataset_from_directory()`

The images are resized to:

`224 × 224`

Pixel values are normalized from:

`[0, 255] → [0, 1]`

The input pipeline is optimized using TensorFlow `AUTOTUNE` and `prefetch()`.

---

## 2. Data Augmentation

Data augmentation is applied only to the training data.

The augmentation pipeline includes:

- Random horizontal flip
- Random rotation
- Random zoom
- Random translation

Data augmentation helps increase the diversity of training images and improves model generalization.

---

# 🧠 Models

## 3. Custom CNN

A CNN model was built from scratch for binary classification.

The architecture includes:

- Data augmentation
- Convolutional layers
- Batch Normalization
- ReLU activation
- MaxPooling
- Dropout
- Global Average Pooling
- Dense layer
- L2 regularization
- Sigmoid output

The model uses:

- Adam optimizer
- Binary cross-entropy loss
- Early stopping
- Learning-rate reduction

---

## 4. EfficientNetB0 Feature Extraction

EfficientNetB0 pretrained on **ImageNet** is used as a feature extractor.

The pretrained backbone is frozen, and a new classification head is added.

### Structure

```text
Input Image
     ↓
EfficientNetB0 (Frozen)
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
Sigmoid Output
```
## 5. ResNet50 Full Fine-Tuning

ResNet50 pretrained on **ImageNet** is used for full fine-tuning.

Unlike feature extraction, the entire ResNet50 backbone is unfrozen and trained together with the new classification head.

### Structure

```text
Input Image
     ↓
Data Augmentation
     ↓
ResNet50 (ImageNet Pretrained)
     ↓
Global Average Pooling
     ↓
Dense Layer (128)
     ↓
Dropout (0.5)
     ↓
Sigmoid Output
```

---

## 📊 Results & Model Comparison

The three models were evaluated on the independent test dataset using accuracy, precision, recall, and F1-score.

| Model | Total Parameters | Test Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 609,153 | 85.06% | 85.76% | 84.08% | 84.91% |
| EfficientNetB0 Feature Extraction | 4,213,668 | **99.28%** | **99.40%** | **99.16%** | **99.28%** |
| ResNet50 Full Fine-Tuning | 23,850,113 | 98.46% | 97.83% | 99.12% | 98.47% |

### 🏆 Best Performing Model

**EfficientNetB0 Feature Extraction** achieved the best overall performance:

- Test Accuracy: **99.28%**
- Precision: **99.40%**
- Recall: **99.16%**
- F1-Score: **99.28%**

### ⚡ Most Parameter-Efficient Model

The **Custom CNN** has the lowest number of parameters with **609,153**, making it the most parameter-efficient model.

Overall, **EfficientNetB0 Feature Extraction** achieved the best overall classification performance among the three models.

## 🔎 Error Analysis

Error analysis was performed on the ResNet50 Fine-Tuned model.

The classification report achieved:

- Accuracy: 98.46%
- Incorrect predictions: 77 out of 5,000 test images

### Confusion Matrix

| Actual / Predicted | Cats | Dogs |
|---|---:|---:|
| Cats | 2445 | 55 |
| Dogs | 22 | 2478 |

The model correctly classified most cats and dogs, with relatively few
misclassifications.
