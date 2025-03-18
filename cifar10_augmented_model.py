import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


#Load the CIFAR-10 Dataset

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize images to [0,1] range
train_images, test_images = train_images / 255.0, test_images / 255.0


#Apply Data Augmentation

data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.2,
    zoom_range=0.6,
    horizontal_flip=True
)

# Augment training images
train_generator = data_gen.flow(train_images, train_labels, batch_size=64)


#Visualize Augmented Images

def display_augmented_images(data_gen, images):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    for i in range(5):  # Show 5 images
        img = np.expand_dims(images[i], axis=0)  # Expand dims for generator
        aug_iter = data_gen.flow(img, batch_size=1)
        aug_img = next(aug_iter)[0]  # Get augmented image

        # Show original
        axes[0, i].imshow(images[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Show augmented
        axes[1, i].imshow(aug_img)
        axes[1, i].set_title("Augmented")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

# Display augmentation effects
display_augmented_images(data_gen, train_images[:5])


#Load & Train the Model

# Load your baseline model
model = load_model("cifar10_baseline_model.keras")

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

# Save the new trained model
model.save("cifar10_augmented_model.keras")


#Plot Training Performance

def plot_metrics(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

plot_metrics(history)
