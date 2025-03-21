# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Import train_test_split

# Load CIFAR-10 dataset
(x_data, y_data), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values from range [0, 255] to [0, 1]
x_data = x_data / 255.0
x_test = x_test / 255.0

# Class names in CIFAR-10 dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Visualize some training images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_data[i])
    plt.xlabel(class_names[y_data[i][0]])
plt.show()

# Split the dataset into training (70%) and validation (30%)
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42
)

# Define the Convolutional Neural Network (CNN) model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),

    # Output layer for 10 classes with softmax activation
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Include accuracy metric

# Display the model summary
model.summary()

# Train the model for 50 epochs with batch size of 255
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=255,
    validation_data=(x_val, y_val)
)

# ✅ Save the trained model to a file
model.save("cifar10_cnn_model.keras")

# Optional: Plot training vs validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
