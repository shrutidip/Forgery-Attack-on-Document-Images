# Fake Image Detection using ELA and CNN

## Project Overview
This project aims to detect fake images using Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs) implemented in Keras. It utilizes ELA to analyze image artifacts introduced during editing, which are then classified using a CNN model.

## Dependencies
- Python 3
- Keras
- numpy
- pandas
- matplotlib
- seaborn
- PIL

## Data Preparation
- The dataset (`dataset_FakeImageDetector_2.csv`) contains paths to images categorized as real or fake.
- Images are processed using ELA to highlight differences in compression levels between original and edited versions.
- Pixel values are normalized to [0, 1] and resized to 128x128 pixels for compatibility with the CNN model.

## Model Architecture
- The CNN architecture consists of convolutional layers (`Conv2D`), max pooling (`MaxPooling2D`), dropout layers (`Dropout`), and dense layers (`Dense`).
- Input Layer: 128x128x3 pixels
- Output Layer: 2 classes (real or fake) with softmax activation for binary classification.

## Training
- Optimizer: RMSprop with a learning rate of 0.0005, rho=0.9, epsilon=1e-08, and no decay.
- Loss Function: Categorical Cross-Entropy
- Early Stopping: Monitoring validation accuracy with a patience of 2 epochs to prevent overfitting.

## Results
- The model achieves an accuracy of X% on the validation set after Y epochs.
- Training history including loss and accuracy metrics can be visualized to understand model performance.
