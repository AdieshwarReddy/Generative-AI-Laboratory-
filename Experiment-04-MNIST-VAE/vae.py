
# Generative AI Laboratory
# Experiment 4: Variational Autoencoder on MNIST

import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------
# Step 1: Configuration
# -----------------------------------

SEED = 42
LATENT_DIM = 2
EPOCHS = 10
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------------
# Step 2: Load MNIST dataset
# -----------------------------------

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

print("GENERATIVE AI LAB - EXPERIMENT 4")
print("Variational Autoencoder on MNIST")

print("\nTraining Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# -----------------------------------
# Step 3: Reparameterization Layer
# -----------------------------------

@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):

    def call(self, inputs):

        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return (
            z_mean +
            tf.exp(0.5 * z_log_var) * epsilon
        )


# -----------------------------------
# Step 4: Build Encoder
# -----------------------------------

encoder_inputs = keras.Input(
    shape=(28, 28, 1)
)

x = layers.Flatten()(encoder_inputs)

x = layers.Dense(
    256,
    activation="relu"
)(x)

z_mean = layers.Dense(
    LATENT_DIM,
    name="z_mean"
)(x)

z_log_var = layers.Dense(
    LATENT_DIM,
    name="z_log_var"
)(x)

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(
    encoder_inputs,
    [z_mean, z_log_var, z],
    name="encoder"
)


# -----------------------------------
# Step 5: Build Decoder
# -----------------------------------

latent_inputs = keras.Input(
    shape=(LATENT_DIM,)
)

x = layers.Dense(
    256,
    activation="relu"
)(latent_inputs)

x = layers.Dense(
    784,
    activation="sigmoid"
)(x)

decoder_outputs = layers.Reshape(
    (28, 28, 1)
)(x)

decoder = keras.Model(
    latent_inputs,
    decoder_outputs,
    name="decoder"
)


# -----------------------------------
# Step 6: Define VAE Model
# -----------------------------------

class VAE(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):

        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def compute_losses(self, data, training=False):

        z_mean, z_log_var, z = self.encoder(
            data,
            training=training
        )

        reconstruction = self.decoder(
            z,
            training=training
        )

        # Binary cross-entropy per pixel
        bce = keras.losses.binary_crossentropy(
            data,
            reconstruction
        )

        # Sum over spatial dimensions
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(bce, axis=(1, 2))
        )

        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var),
                axis=1
            )
        )

        total_loss = reconstruction_loss + kl_loss

        return (
            total_loss,
            reconstruction_loss,
            kl_loss
        )

    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            total_loss, recon_loss, kl_loss = (
                self.compute_losses(
                    data,
                    training=True
                )
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            recon_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "total_loss":
                self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result()
        }

    def test_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        total_loss, recon_loss, kl_loss = (
            self.compute_losses(
                data,
                training=False
            )
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            recon_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "total_loss":
                self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result()
        }


# -----------------------------------
# Step 7: Compile VAE
# -----------------------------------

vae = VAE(
    encoder,
    decoder,
    name="MNIST_VAE"
)

vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    )
)

print("\nEncoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()


# -----------------------------------
# Step 8: Train VAE
# -----------------------------------

history = vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test,),
    shuffle=True,
    verbose=1
)


# -----------------------------------
# Step 9: Evaluate VAE
# -----------------------------------

evaluation = vae.evaluate(
    x_test,
    batch_size=BATCH_SIZE,
    return_dict=True,
    verbose=0
)

print("\nFinal Evaluation:")

for metric, value in evaluation.items():
    print(f"{metric}: {value:.4f}")


# -----------------------------------
# Step 10: Reconstruct Images
# -----------------------------------

z_mean, _, _ = encoder.predict(
    x_test[:10],
    verbose=0
)

reconstructed = decoder.predict(
    z_mean,
    verbose=0
)

plt.figure(figsize=(15, 4))

for i in range(10):

    plt.subplot(2, 10, i + 1)

    plt.imshow(
        x_test[i].squeeze(),
        cmap="gray"
    )

    plt.axis("off")

    if i == 0:
        plt.ylabel("Original")

    plt.subplot(2, 10, i + 11)

    plt.imshow(
        reconstructed[i].squeeze(),
        cmap="gray"
    )

    plt.axis("off")

    if i == 0:
        plt.ylabel("Rebuilt")

plt.suptitle(
    "MNIST VAE - Original vs Reconstructed Images"
)

plt.tight_layout()

plt.savefig(
    "outputs/reconstructed_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------
# Step 11: Generate New Digit Images
# -----------------------------------

grid_size = 10

grid = np.zeros(
    (28 * grid_size, 28 * grid_size)
)

latent_values = np.linspace(
    -2.5,
    2.5,
    grid_size
)

for i, y in enumerate(latent_values):

    for j, x in enumerate(latent_values):

        latent_vector = np.array(
            [[x, y]],
            dtype="float32"
        )

        generated = decoder.predict(
            latent_vector,
            verbose=0
        )

        digit = generated[0].squeeze()

        grid[
            i * 28:(i + 1) * 28,
            j * 28:(j + 1) * 28
        ] = digit

plt.figure(figsize=(10, 10))

plt.imshow(
    grid,
    cmap="gray"
)

plt.title(
    "New MNIST Digits Generated from Latent Space"
)

plt.axis("off")

plt.savefig(
    "outputs/generated_digits.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------
# Step 12: Plot Training Loss
# -----------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["total_loss"],
    label="Training Total Loss"
)

plt.plot(
    history.history["val_total_loss"],
    label="Validation Total Loss"
)

plt.plot(
    history.history["reconstruction_loss"],
    label="Reconstruction Loss"
)

plt.plot(
    history.history["kl_loss"],
    label="KL Divergence Loss"
)

plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "outputs/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------
# Step 13: Save Models
# -----------------------------------

encoder.save(
    "models/vae_encoder.keras"
)

decoder.save(
    "models/vae_decoder.keras"
)

print("\nGenerated Files:")

print("outputs/reconstructed_images.png")
print("outputs/generated_digits.png")
print("outputs/training_loss.png")

print("models/vae_encoder.keras")
print("models/vae_decoder.keras")

print("\nExperiment 4 completed successfully!")

/*  Important concepts used in the code
Component	Purpose
z_mean	Mean of the latent Gaussian distribution
z_log_var	Logarithm of the latent variance
Sampling	Implements the reparameterization trick
tf.random.normal()	Generates Gaussian noise
encoder	Converts images into latent distributions
decoder	Converts latent vectors into images
GradientTape()	Computes gradients during training
binary_crossentropy	Measures reconstruction error
kl_loss	Regularizes the latent distribution
decoder.predict()	Generates images from latent vectors */
