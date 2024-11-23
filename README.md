# Music-Gerre-Classification Using Spectrograms and CNN 

## Project Description
This project focuses on using machile learning skills to accomplish music genre classification. By converting audio files into spectrograms we are able to use Convolutional Neural Networks (CNNs) on audio spectrograms. This allows CNN to highlight key features associated to different genres in order to classify them. That allows us to accmoplish our main objective which is to give the model a music sample and to corretly classify which genre ti belongs to.

## Key goals
- Develop a CNN model that uses spectrograms for music genre classification
- Use the GTZAN dataset for audio clips to turn into spectrograms
- Achieve a good classifcation accuracy using the CNN model

## Table of Contents
1. [Project Details](#project-details)
2. [Technologies Used](#technologies-used)
3. [Requirments needed](#requirements-used)
4. [Code Examples](#code-examples)
5. [Contribution Guidelines](#contribution-guidelines)
6. [Acknowledgments](#acknowledgments)

## Project Details

- **Purpose**: The goal of this project is to create an efficient music genre classifier using audio data. By converting raw audio into spectrograms, we can insert these images into a Convolutional Neural Network for classification.
  
- **Key Features**:
  - Data preprocessing: Converts audio files into spectrogram images.
  - Data augmentation: Increases model robustness through various transformations.
  - Model: A custom CNN built using the ResNet50V2 architecture.
  - Evaluation: Accuracy and confusion matrix to assess model performance.

## Technologies Used

- **Python**: The primary programming language used.
- **TensorFlow/Keras**: For model building and training.
- **Librosa**: For audio processing and spectrogram generation.
- **Matplotlib**: For visualization and plotting.
- **Kaggle API**: For downloading the GTZAN dataset.

## Requirements:
- Python 3.x
- TensorFlow 2.x
- Librosa
- Matplotlib

## Code examples 

-Model Training example

Define the model architecture
model = create_model((224, 224, 3), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Fit the model with data augmentation
history = model.fit(train_data, validation_data=val_data, epochs=40, callbacks=callbacks)

-Spectragram Generation example 

import librosa
import librosa.display
import matplotlib.pyplot as plt

def save_spectrogram(y, sr, save_path):
    plt.figure(figsize=(2, 2))  
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

## Contribution Guidelines 

1. Fork the repository.
2. Clone your fork and create a new branch for your contribution:  git checkout -b feature-branch
3. Make your changes and commit them: ex. git commit -m "Add new feature or fix bug"
4. Push your changes: git push origin feature-branch

## Acknowledgements 

1. GTZAN Dataset: Used the GTZAN music genre dataset provided by GTZAN Music Genre Dataset.
2. TensorFlow/Keras: For building and training the deep learning model.
3. Librosa: For audio processing and spectrogram generation.



