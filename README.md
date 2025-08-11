# Image-Recognition-model
This project explores image classification using deep learning with TensorFlow and Keras, applying techniques across multiple datasets to demonstrate model development, optimization, and evaluation.  We begin with the MNIST dataset of handwritten digits, implementing a simple convolutional neural network (CNN) from scratch. 
Image Classification Project

# Image Classification Project

This project explores image classification using various datasets and deep learning models in TensorFlow and Keras.

## Project Goals:

*   Learn and implement image classification techniques.
*   Experiment with different datasets (MNIST, CIFAR-10, Cats vs. Dogs).
*   Build and train Convolutional Neural Networks (CNNs) from scratch.
*   Utilize transfer learning with a pre-trained model (MobileNetV2).
*   Evaluate model performance using metrics like accuracy, confusion matrix, and ROC curves.

## Notebook Steps:

1.  **Setup and Data Download:** Install necessary libraries and download datasets using Kaggle API and TensorFlow built-in datasets.
2.  **MNIST Classification:**
    *   Load and preprocess the MNIST dataset.
    *   Build a simple CNN model.
    *   Train and evaluate the MNIST model.
    *   Visualize training history and test accuracy.
3.  **CIFAR-10 Classification:**
    *   Load and preprocess the CIFAR-10 dataset.
    *   Implement data augmentation using `ImageDataGenerator`.
    *   Build a more complex CNN model with Batch Normalization and Dropout.
    *   Train and evaluate the CIFAR-10 model with data augmentation.
    *   Analyze model performance using a confusion matrix and classification report.
4.  **Cats vs. Dogs Classification (Transfer Learning):**
    *   Download and extract the Cats vs. Dogs dataset.
    *   Set up `ImageDataGenerator` for training and validation data with rescaling and validation split.
    *   Load a pre-trained MobileNetV2 model and add custom classification layers.
    *   Train the model with frozen base layers (transfer learning).
    *   Fine-tune the model by making the base layers trainable with a low learning rate.
    *   Save and load the trained model.
    *   Evaluate the model using an ROC curve and AUC.
    *   Demonstrate how to make predictions on new images.
5.  **Performance Comparison:** Visualize the accuracies of the models trained on the different datasets.

## Technologies Used:

*   Python
*   TensorFlow
*   Keras
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   Kaggle API
*   KaggleHub
Scikit-learn
Kaggle API
KaggleHub
