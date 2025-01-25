# Aura Sense - Gender and Mood Recognizer System

Aura Sense is an advanced machine learning project designed to classify gender and recognize moods in real time using deep learning techniques. The system integrates computer vision methodologies and hybrid machine learning approaches to achieve robust results in human attribute recognition. With applications spanning mental health care, human-computer interaction, and personalized experiences, Aura Sense is a step forward in making AI more empathetic and adaptive.

## Table of Contents
- [Motivation](#motivation)
- [Project Objectives](#project-objectives)
- [Significance](#significance)
- [Key Features](#key-features)
- [Dataset](#dataset)
  - [Gender Recognition](#gender-recognition)
  - [Mood Recognition](#mood-recognition)
- [Model Architectures](#model-architectures)
  - [Gender Recognition Model](#gender-recognition-model)
  - [Mood Recognition Model](#mood-recognition-model)
- [Training Strategy](#training-strategy)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [Future Work](#future-work)
- [Project Attachments](#project-attachments)

---

## Motivation

In the era of advanced AI, interpreting human attributes has become crucial for improving user experiences and enabling adaptive systems. Gender recognition and mood detection have applications in psychological research, security systems, and accessibility tools. The Aura Sense project aims to bridge the gap between AI capabilities and human understanding.

## Project Objectives

- Develop a robust gender recognition system using facial image analysis.
- Build a real-time mood detection model for identifying emotional states.
- Explore multi-modal integration techniques for enhanced human attribute recognition.

## Significance

By combining facial and potential audio modalities, this project demonstrates the possibilities of hybrid machine learning systems in various fields:

- **Human-computer interaction**
- **Psychological assessment tools**
- **Personalized digital experiences**
- **Accessibility technologies**

## Key Features

- Real-time gender classification (Male/Female).
- Detection of seven emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- High-performance models leveraging transfer learning and custom architectures.
- Flask-based web application for live video testing and demos.

## Dataset

### Gender Recognition

- **Dataset**: CelebA dataset
- **Preprocessing**:
  - Standardized images to 128x128 pixels.
  - Normalized pixel values to the 0-1 range.

### Mood Recognition

- **Dataset**: Facial Expression Recognition Challenge 2013
- **Preprocessing**:
  - Grayscale images resized to 48x48 pixels.
  - Face detection and automatic cropping using OpenCV.

## Model Architectures

### Gender Recognition Model

- **Base Model**: MobileNetV2 (pre-trained on ImageNet).
- **Additional Layers**:
  - GlobalAveragePooling2D layer.
  - Dense layers with ReLU activations and Dropout for regularization.
  - Final Dense layer with sigmoid activation for binary classification.

### Mood Recognition Model

- **Architecture**: Convolutional Neural Network (CNN).
- **Layer Configuration**:
  - Multiple Conv2D layers with increasing filter sizes.
  - BatchNormalization, ReLU activations, and MaxPooling2D layers.
  - Dense layers with softmax activation for multi-class classification (7 moods).

## Training Strategy

- **Framework**: TensorFlow & Keras
- **Gender Recognition**:
  - Optimizer: Adam
  - Learning Rate: 0.0005
  - Epochs: 15
  - Batch Size: 64
- **Mood Recognition**:
  - Optimizer: Adam
  - Learning Rate: 0.0005
  - Epochs: 15
  - Batch Size: 64
  - ReduceLROnPlateau and ModelCheckpoint callbacks for dynamic adjustments.

## Performance Metrics

### Gender Recognition
- Training Accuracy: 99.54% - 99.70%
- Validation Accuracy: 98.23% - 98.42%
- Test Accuracy: 98.15%

### Mood Recognition
- Training Accuracy: 53.1% - 67.2%
- Validation Accuracy: 59.4% - 61.8%
- Real-time testing demonstrated better performance due to robust temporal consistency and feature extraction.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, Flask
- **Tools**: Jupyter Notebook

## Contributors

- **Sachet Utekar**: Gender recognition implementation, dataset research, YouTube demo.
- **Shubham Doshi**: Dataset research, initial training, testing, project management, report documentation.
- **Neeraj Saini**: Mood recognition dataset and model research,
Mood recognition model implementation, final presentation documentation.

## Future Work

- Enhance mood recognition accuracy through multi-modal approaches (e.g., audio and text inputs).
- Deploy the system on resource-constrained devices.
- Explore ensemble models for improved robustness.

---

## Project Attachments

- **Presentation Link**: [https://www.youtube.com/watch?v=BnlZIYyMLEc](https://www.youtube.com/watch?v=BnlZIYyMLEc)
- **Project Demo Link**: [https://www.youtube.com/watch?v=uZQCt3bLOB4](https://www.youtube.com/watch?v=uZQCt3bLOB4)
- **Dataset Link**: [https://drive.google.com/drive/folders/1CKlh-vGU9zvo0ecUSl-6a2an2NNx0KSs?usp=drive_link](https://drive.google.com/drive/folders/1CKlh-vGU9zvo0ecUSl-6a2an2NNx0KSs?usp=drive_link)
- **Project Documents Link**: [https://drive.google.com/drive/folders/1qzruPMLvshQps-Ra_jqSF_CUeG1VGYIG?usp=drive_link](https://drive.google.com/drive/folders/1qzruPMLvshQps-Ra_jqSF_CUeG1VGYIG?usp=drive_link)
