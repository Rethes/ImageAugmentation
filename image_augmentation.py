import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Select 5 random images and normalize them
random_indices = random.sample(range(len(train_images)), 5)
sample_images = train_images[random_indices].astype('float32') / 255.0  # Normalize to [0,1]

# Define different augmentation techniques separately
rotation_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40)
width_shift_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.4)
height_shift_gen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.2)
zoom_gen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.6)
flip_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)


# Function to display images with specific augmentations
def display_augmented_images():
    plt.figure(figsize=(12, 15))

    for i, img in enumerate(sample_images):
        img = np.expand_dims(img, axis=0)  # Expand dims to match batch input format

        # Generate augmented images
        rotated_img = next(rotation_gen.flow(img, batch_size=1))[0]
        width_shifted_img = next(width_shift_gen.flow(img, batch_size=1))[0]
        height_shifted_img = next(height_shift_gen.flow(img, batch_size=1))[0]
        zoomed_img = next(zoom_gen.flow(img, batch_size=1))[0]
        flipped_img = next(flip_gen.flow(img, batch_size=1))[0]

        # Display original image
        plt.subplot(6, 5, i + 1)
        plt.imshow(img[0])
        plt.title("Original")
        plt.axis("off")

        # Display rotated image
        plt.subplot(6, 5, i + 6)
        plt.imshow(rotated_img)
        plt.title("Rotated")
        plt.axis("off")

        # Display width-shifted image
        plt.subplot(6, 5, i + 11)
        plt.imshow(width_shifted_img)
        plt.title("Width Shifted")
        plt.axis("off")

        # Display height-shifted image
        plt.subplot(6, 5, i + 16)
        plt.imshow(height_shifted_img)
        plt.title("Height Shifted")
        plt.axis("off")

        # Display zoomed image
        plt.subplot(6, 5, i + 21)
        plt.imshow(zoomed_img)
        plt.title("Zoomed")
        plt.axis("off")

        # Display flipped image
        plt.subplot(6, 5, i + 26)
        plt.imshow(flipped_img)
        plt.title("Flipped")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Display images with specific augmentations
display_augmented_images()
