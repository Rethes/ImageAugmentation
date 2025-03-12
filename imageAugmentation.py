import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# Define the ImageDataGenerator with augmentation techniques
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,  # Rotate images up to 40 degrees
    width_shift_range=0.4,  # Shift width by 20%
    height_shift_range=0.2,  # Shift height by 20%
    zoom_range=0.2,  # Random zoom by 20%
    horizontal_flip=True,  # Flip images horizontally
    rescale=1.0 / 255  # Normalize pixel values
)

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Select 5 random images for augmentation demonstration
random_indices = random.sample(range(len(train_images)), 5)
sample_images = train_images[random_indices]

# Display original and augmented images
def display_images():
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(sample_images):
        img = np.expand_dims(img, axis=0)  # Expand dims to match batch input format
        aug_iter = data_gen.flow(img, batch_size=1)
        augmented_image = next(aug_iter)[0]  # Get next augmented image

        # Show original image
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(img[0])
        plt.title("Original")
        plt.axis("off")

        # Show augmented image
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(augmented_image)
        plt.title("Augmented")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Display the images
display_images()