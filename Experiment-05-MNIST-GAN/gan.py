
# Generative AI Laboratory
# Experiment 5: GAN on MNIST

import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# -------------------------------------
# Step 1: Configuration
# -------------------------------------

SEED = 42
NOISE_DIM = 100
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.0002

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -------------------------------------
# Step 2: Load MNIST Dataset
# -------------------------------------

(x_train, _), (_, _) = (
    keras.datasets.mnist.load_data()
)

print("GENERATIVE AI LAB - EXPERIMENT 5")
print("Basic GAN on MNIST")

print("\nOriginal Dataset Shape:", x_train.shape)

# Normalize pixel values to [-1, 1]
x_train = (
    x_train.astype("float32") - 127.5
) / 127.5

x_train = np.expand_dims(
    x_train,
    axis=-1
)

print("Processed Dataset Shape:", x_train.shape)


# -------------------------------------
# Step 3: Create TensorFlow Dataset
# -------------------------------------

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(60000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# -------------------------------------
# Step 4: Build Generator
# -------------------------------------

def build_generator():

    model = keras.Sequential(
        [
            layers.Input(shape=(NOISE_DIM,)),

            layers.Dense(
                256,
                activation="relu"
            ),

            layers.Dense(
                512,
                activation="relu"
            ),

            layers.Dense(
                1024,
                activation="relu"
            ),

            layers.Dense(
                784,
                activation="tanh"
            ),

            layers.Reshape((28, 28, 1))
        ],
        name="Generator"
    )

    return model


# -------------------------------------
# Step 5: Build Discriminator
# -------------------------------------

def build_discriminator():

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            layers.Flatten(),

            layers.Dense(512),
            layers.LeakyReLU(
                negative_slope=0.2
            ),

            layers.Dense(256),
            layers.LeakyReLU(
                negative_slope=0.2
            ),

            layers.Dense(
                1,
                activation="sigmoid"
            )
        ],
        name="Discriminator"
    )

    return model


generator = build_generator()
discriminator = build_discriminator()

print("\nGenerator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()


# -------------------------------------
# Step 6: Define Loss Function
# -------------------------------------

cross_entropy = (
    keras.losses.BinaryCrossentropy()
)


def discriminator_loss(
    real_output,
    fake_output
):

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


# -------------------------------------
# Step 7: Define Optimizers
# -------------------------------------

generator_optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)


# -------------------------------------
# Step 8: Training Step
# -------------------------------------

@tf.function
def train_step(real_images):

    batch_size = tf.shape(real_images)[0]

    noise = tf.random.normal(
        [batch_size, NOISE_DIM]
    )

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        # Generate fake images
        fake_images = generator(
            noise,
            training=True
        )

        # Discriminator predictions
        real_output = discriminator(
            real_images,
            training=True
        )

        fake_output = discriminator(
            fake_images,
            training=True
        )

        # Calculate losses
        gen_loss = generator_loss(
            fake_output
        )

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    # Calculate gradients
    generator_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    # Update networks
    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss


# -------------------------------------
# Step 9: Save Generated Images
# -------------------------------------

fixed_noise = tf.random.normal(
    [16, NOISE_DIM],
    seed=SEED
)


def save_generated_images(epoch, filename=None):

    predictions = generator(
        fixed_noise,
        training=False
    ).numpy()

    # Convert [-1, 1] to [0, 1]
    predictions = (
        predictions + 1.0
    ) / 2.0

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(6, 6)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            predictions[i, :, :, 0],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.axis("off")

    fig.suptitle(
        f"GAN Generated Images - Epoch {epoch}"
    )

    plt.tight_layout()

    if filename is None:
        filename = (
            f"outputs/generated_epoch_{epoch:02d}.png"
        )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------------------
# Step 10: Train GAN
# -------------------------------------

generator_losses = []
discriminator_losses = []

print("\nStarting GAN Training...\n")

for epoch in range(1, EPOCHS + 1):

    gen_loss_metric = keras.metrics.Mean()
    disc_loss_metric = keras.metrics.Mean()

    for real_images in train_dataset:

        gen_loss, disc_loss = train_step(
            real_images
        )

        gen_loss_metric.update_state(
            gen_loss
        )

        disc_loss_metric.update_state(
            disc_loss
        )

    avg_gen_loss = float(
        gen_loss_metric.result().numpy()
    )

    avg_disc_loss = float(
        disc_loss_metric.result().numpy()
    )

    generator_losses.append(
        avg_gen_loss
    )

    discriminator_losses.append(
        avg_disc_loss
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Generator Loss: {avg_gen_loss:.4f} | "
        f"Discriminator Loss: {avg_disc_loss:.4f}"
    )

    if epoch in [1, 5, 10, 20]:

        save_generated_images(epoch)


# -------------------------------------
# Step 11: Save Final Images
# -------------------------------------

save_generated_images(
    EPOCHS,
    filename="outputs/generated_digits.png"
)


# -------------------------------------
# Step 12: Plot Training Loss
# -------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")

plt.title(
    "GAN Generator and Discriminator Training Loss"
)

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "outputs/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------
# Step 13: Save Models
# -------------------------------------

generator.save(
    "models/gan_generator.keras"
)

discriminator.save(
    "models/gan_discriminator.keras"
)


# -------------------------------------
# Step 14: Final Output
# -------------------------------------

print("\nGenerated Files:")

print("outputs/generated_epoch_01.png")
print("outputs/generated_epoch_05.png")
print("outputs/generated_epoch_10.png")
print("outputs/generated_epoch_20.png")
print("outputs/generated_digits.png")
print("outputs/training_loss.png")

print("models/gan_generator.keras")
print("models/gan_discriminator.keras")

print("\nExperiment 5 completed successfully!")
