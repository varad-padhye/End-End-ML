# 🌿 Leaf Disease Detection Using Machine Learning and Deep Learning

## Overview

This repository presents a complete image classification pipeline for **leaf disease detection**, covering both **traditional machine learning** and **deep learning approaches**.

The dataset consists of **leaf images belonging to 36 classes**, including multiple disease categories and healthy leaves.  
The project follows a progressive modeling strategy, starting with exploratory data analysis and classical machine learning, and advancing to convolutional neural networks and transfer learning.

---


---

## Dataset Description

- Input: RGB leaf images  
- Image size: Resized to 224 × 224  
- Total classes: 36  
- Labels: Diseased and healthy leaf categories  
- Key challenge: Class imbalance across disease types  

---

## Part 1: Exploratory Data Analysis and Classical Machine Learning

File: `EDA.ipynb`
File: `analysis-using-ml.ipynb`


### Workflow

- Dataset inspection and visualization  
- Class distribution and imbalance analysis  
- Image feature extraction:
  - Color histograms  
  - Color moments  
  - Texture based features  
- Feature scaling and preprocessing  
- Training and evaluation of a Random Forest classifier  

### Motivation for Random Forest

- Strong baseline for structured features  
- Robust to noise and overfitting  
- Provides feature importance  
- Helps identify limitations of handcrafted features  

### Observations

- Majority classes dominate predictions  
- Minority disease classes show low recall  
- Performance plateaus due to limited feature expressiveness  
- Justifies transition to deep learning  

---

## Part 2: Deep Learning Pipeline Using PyTorch

File: `deep_learning_pytorch.py`

### Data Preprocessing

- Image resizing to 224 × 224  
- Normalization using ImageNet statistics  
- Data augmentation:
  - Random rotation  
  - Horizontal flip  
  - Random resized crop  

---

### Handling Class Imbalance

- WeightedRandomSampler used during training  
- Ensures balanced sampling across all classes  
- Prevents majority class bias without data duplication  

---

### Model A: Baseline Convolutional Neural Network

- Two convolution blocks  
- ReLU activation and max pooling  
- Fully connected classifier  
- Used as a deep learning baseline  

---

### Model B: Transfer Learning

- Backbone: MobileNetV2 pretrained on ImageNet  
- Frozen feature extractor  
- Custom classification head  
- Fine tuning of last layers for domain adaptation  

---

### Evaluation Metrics

- Overall accuracy  
- Per class metrics:
  - Precision  
  - Recall  
  - F1 score  
  - Support  
- Confusion matrix for detailed error analysis  

Metrics are reported individually for all 36 classes.

---

## Results Summary

| Model            | Strengths                              | Limitations                                |
|------------------|----------------------------------------|--------------------------------------------|
| Random Forest    | Interpretable baseline                 | Weak on visually similar diseases           |
| Simple CNN       | Learns spatial features                | Limited generalization                      |
| Transfer Learning| Best overall accuracy and recall       | Confusion among similar disease patterns   |

---

## Key Insights

- Transfer learning significantly improves minority class recall  
- Pretrained models capture leaf texture and disease patterns effectively  
- Classical machine learning struggles with complex visual variability  
- Class imbalance strongly affects disease classification performance  

---

## Future Improvements

- Focal loss for hard samples  
- Grad CAM for explainability  
- Vision Transformer based models  
- More samples for rare disease classes  
- Deployment using FastAPI or Streamlit  

---

## Requirements

-torch
-torchvision
-numpy
-scikit-learn
-matplotlib
-seaborn




