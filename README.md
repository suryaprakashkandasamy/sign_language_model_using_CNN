# Sign Language Detection Model

## Overview
This project implements a convolutional neural network (CNN) using TensorFlow and Keras for recognizing letters from American Sign Language (ASL) gestures. The model is trained on the Sign Language MNIST dataset, consisting of grayscale images of hand gestures representing 24 classes (letters A-Z, excluding J and Z which involve motion).

## Requirements

### Required Libraries
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Additional Libraries Used
- `warnings`: Suppress specific warnings.
- `LabelBinarizer` from `sklearn.preprocessing`: For label encoding.
- `ImageDataGenerator` from `tensorflow.keras.preprocessing.image`: Used for data augmentation.
- `Regularizers` from `tensorflow.keras`: Regularization techniques for model building.

## Project Structure
- `sign_language_model_using_CNN.ipynb`: Jupyter Notebook containing complete code for data preprocessing, model building, training, evaluation, and prediction.
- `sign_language_model_using_CNN.h5`: Trained model saved in H5 file format.

## Customization

### Model Architecture
Modify the CNN architecture in the notebook (`sign_language_model_using_CNN.ipynb`) to experiment with different configurations.

### Hyperparameter Tuning
Adjust hyperparameters directly in the notebook for optimal model performance.

### Data Augmentation
Explore different augmentation techniques by modifying the `ImageDataGenerator` parameters.
