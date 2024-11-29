# Deep Learning Coursework ReadMe

## Overview
This repository contains Python code and project files related to a deep learning course. It includes two lab exercises and one project focusing on medical image classification and processing using deep learning techniques. The coursework covers tasks such as pneumonia detection using X-rays and lung cancer classification from CT images, employing convolutional neural networks (CNNs) and advanced preprocessing techniques.

## Files and Folders
### 1. Lab 1: Pneumonia Detection with TPU Acceleration
  * **Objectives:** Classify chest X-ray images into ```NORMAL``` and ```PNEUMONIA``` categories.
  * **Key Highlights:**
      * Utilizes distributed training on TPU for enhanced performance.
      * Implements data preprocessing using TensorFlow datasets.
      * Includes a custom CNN model built using Keras with Conv2D, SeparableConv2D, and dense layers.
      * Handles class imbalance with calculated class weights and initial biases.
      * Tracks metrics including precision, recall, and binary accuracy during training.
      * Visualizes training history with plots and evaluates model performance on test data.
  * **Dependencies:**
      * Tensorflow
      * Keras
      * Matplotlib
      * NumPy
   
### 2. Lab 2: Lung Cancer Detection using Convolutional Neural Networks
  * **Objectives:** Classify lung cancer cases into ```Benign```, ```Malignant```, and ```Normal``` using the IQ-OTHNCCD dataset.
  * **Key Highlights:**
      * Performs data augmentation with histogram equalization and resizing.
      * Addresses class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
      * Builds a CNN model with layers for convolution, pooling, and fully connected layers.
      * Trains the model with calculated class weights to mitigate class imbalance.
      * Evaluates performance using metrics such as accuracy, recall, precision, and F1-score.
  * **Dependencies:**
      * Tensorflow
      * Keras
      * Matplotlib
      * NumPy
      * scikit-learn
      * imbalanced-learn
   
### 3. Project: Advanced Medical Image Classification
* **Objectives:** Develop and evaluate advanced deep learning models for medical image classification.
* **Key Highlights:**
  * Combines preprocessing and CNN modelling approaches from both labs.
  * Trains models on real-world medical datasets with varying distributions and complexities.
  * Applies best practices in deep learning such as early stopping, model checkpoints, and learning rate scheduling.
  * Provides insights through detailed metrics and visualizations.
