# ImageAugmentation

## Project description
This project demonstrates the use of TensorFlow's ImageDataGenerator to apply various image augmentation techniques to a dataset. The CIFAR-10 dataset is used for demonstration purposes, and five random images are selected to show their original and augmented versions.It also focuses on training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 images across 10 classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Why the CIFAR-10 dataset was used
It has 10 distinct classes
Augmenting low-resolution images forces the model to learn robust features instead of memorizing specific details.
CIFAR-10 is easily accessible in libraries like TensorFlow 


## Installation
1.⁠ ⁠Clone the Git hub repository or download the zip file.
https://github.com/Rethes/ImageAugmentation.git
2.Create Virtual Environment (optional) - on Mac
python -m venv env
source env/bin/activate
3.Install the Dependencies from requirements.txt File
pip install -r requirements.txt
4.Run the Script
python scripts/train.py


## Technologies Used
●	Python 3.7+
●	TensorFlow
●	NumPy
●	Matplotlib

## Features
Image Augmentations: Random horizontal flip, rotation, cropping, color jitter, and Gaussian blur.
CNN Model Training: Trains a simple CNN on both original and augmented datasets.
Performance Evaluation: Compares accuracy and loss curves between models.
Visualizations: Displays augmented images and training progress.

## Image Augmentation Techniques Used
The following augmentation techniques are applied:
●	Rotation: Rotates images up to 40 degrees
●	Width Shift: Shifts the image horizontally by up to 20%
●	Height Shift: Shifts the image vertically by up to 20%
●	Zoom: Random zoom by up to 20%
●	Horizontal Flip: Flips the image horizontally
●	Rescaling: Normalizes pixel values to the range [0,1]

## Training and Evaluation
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. The training dataset is split into training and validation sets, with 80% of the data used for training and 20% used for validation. The model is trained for 10 epochs with a batch size of 64

