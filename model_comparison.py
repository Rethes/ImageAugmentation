# Load the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (convert from 0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the baseline model
baseline_model = load_model("cifar10_baseline_model.keras")

# Load the augmented model
augmented_model = load_model("cifar10_augmented_model.keras")

# Evaluate the baseline model
baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test, y_test)

# Evaluate the augmented model
augmented_loss, augmented_accuracy = augmented_model.evaluate(x_test, y_test)

# Print the results
print("Baseline Model:")
print(f"Loss: {baseline_loss:.4f}, Accuracy: {baseline_accuracy:.4f}")

print("Augmented Model:")
print(f"Loss: {augmented_loss:.4f}, Accuracy: {augmented_accuracy:.4f}")

# Plot a comparison of the two models
plt.bar(["Baseline", "Augmented"], [baseline_accuracy, augmented_accuracy])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

# Load the training history of the two models
import pickle
with open('baseline_history.pkl', 'rb') as f:
    baseline_history = pickle.load(f)

with open('augmented_history.pkl', 'rb') as f:
    augmented_history = pickle.load(f)

# Plot the training and validation accuracy and loss for both models
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(baseline_history['accuracy'], label='Baseline Training Accuracy')
plt.plot(baseline_history['val_accuracy'], label='Baseline Validation Accuracy')
plt.plot(augmented_history['accuracy'], label='Augmented Training Accuracy')
plt.plot(augmented_history['val_accuracy'], label='Augmented Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(baseline_history['loss'], label='Baseline Training Loss')
plt.plot(baseline_history['val_loss'], label='Baseline Validation Loss')
plt.plot(augmented_history['loss'], label='Augmented Training Loss')
plt.plot(augmented_history['val_loss'], label='Augmented Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")

plt.tight_layout()
plt.show()