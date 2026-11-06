
# Generative AI Laboratory
# Experiment 3: MNIST Autoencoder

import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist


# ---------------------------------
# Step 1: Set random seeds
# ---------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ---------------------------------
# Step 2: Load MNIST dataset
# ---------------------------------

(x_train, _), (x_test, _) = mnist.load_data()

print("GENERATIVE AI LAB - EXPERIMENT 3")
print("MNIST Autoencoder Image Reconstruction")

print("\nTraining Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)


# ---------------------------------
# Step 3: Normalize images
# ---------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ---------------------------------
# Step 4: Flatten images
# ---------------------------------

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("\nFlattened Training Shape:", x_train.shape)
print("Flattened Testing Shape:", x_test.shape)


# ---------------------------------
# Step 5: Build Encoder
# ---------------------------------

input_image = layers.Input(shape=(784,))

encoded = layers.Dense(
    128,
    activation="relu"
)(input_image)

latent = layers.Dense(
    32,
    activation="relu",
    name="latent_layer"
)(encoded)


# ---------------------------------
# Step 6: Build Decoder
# ---------------------------------

decoded = layers.Dense(
    128,
    activation="relu"
)(latent)

output_image = layers.Dense(
    784,
    activation="sigmoid"
)(decoded)


# ---------------------------------
# Step 7: Create Autoencoder
# ---------------------------------

autoencoder = Model(
    inputs=input_image,
    outputs=output_image,
    name="MNIST_Autoencoder"
)


# ---------------------------------
# Step 8: Compile Model
# ---------------------------------

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("\nAutoencoder Model Summary:")
autoencoder.summary()


# ---------------------------------
# Step 9: Train Model
# ---------------------------------

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)


# ---------------------------------
# Step 10: Evaluate Model
# ---------------------------------

test_loss = autoencoder.evaluate(
    x_test,
    x_test,
    verbose=0
)

print("\nFinal Test Reconstruction Loss:")
print(round(float(test_loss), 6))


# ---------------------------------
# Step 11: Reconstruct Images
# ---------------------------------

reconstructed = autoencoder.predict(
    x_test[:10],
    verbose=0
)


# ---------------------------------
# Step 12: Visualize Images
# ---------------------------------

plt.figure(figsize=(15, 4))

for i in range(10):

    # Original image
    ax = plt.subplot(2, 10, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title("Original")
    plt.axis("off")

    # Reconstructed image
    ax = plt.subplot(2, 10, i + 11)

    plt.imshow(
        reconstructed[i].reshape(28, 28),
        cmap="gray"
    )

    plt.title("Rebuilt")
    plt.axis("off")


plt.suptitle(
    "MNIST Autoencoder - Original vs Reconstructed Images"
)

plt.tight_layout()

plt.savefig(
    "outputs/reconstructed_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------
# Step 13: Plot Training Loss
# ---------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Autoencoder Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "outputs/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------
# Step 14: Save Model
# ---------------------------------

autoencoder.save(
    "models/mnist_autoencoder.keras"
)

print("\nGenerated Files:")
print("outputs/reconstructed_images.png")
print("outputs/training_loss.png")
print("models/mnist_autoencoder.keras")

print("\nExperiment 3 completed successfully!")
