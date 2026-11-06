
# Experiment 4: Variational Autoencoder (VAE) on MNIST

## Aim

To implement a Variational Autoencoder (VAE) from scratch using TensorFlow/Keras and train it on the MNIST handwritten digit dataset for image reconstruction and generation.

## Objectives

1. Understand the architecture of a Variational Autoencoder.
2. Load and preprocess the MNIST dataset.
3. Implement an Encoder and Decoder.
4. Learn the latent mean and log-variance.
5. Apply the reparameterization trick.
6. Train the VAE using reconstruction and KL divergence losses.
7. Reconstruct handwritten digit images.
8. Generate new digit images by sampling from latent space.
9. Visualize the training losses and generated images.

## Theory

### What is a Variational Autoencoder?

A Variational Autoencoder (VAE) is a probabilistic generative neural network that learns a continuous latent representation of input data.

Unlike a traditional Autoencoder, which maps an input to a fixed latent vector, a VAE learns the parameters of a probability distribution.

The VAE can then sample latent vectors and generate new data.

### Components of a VAE

1. Encoder
2. Latent Mean
3. Latent Log-Variance
4. Reparameterization Layer
5. Decoder

### Encoder

The Encoder converts the input image into two vectors:

- Mean (z_mean)
- Log-variance (z_log_var)

These describe a Gaussian distribution in latent space.

### Reparameterization Trick

Instead of directly sampling z, the VAE uses:

z = μ + exp(0.5 × log(σ²)) × ε

Where:

- μ = Latent mean
- σ² = Latent variance
- ε = Random noise sampled from N(0, I)

This allows gradients to flow through the sampling operation during training.

### Decoder

The Decoder converts a sampled latent vector into a reconstructed image.

### VAE Loss Function

The total VAE loss consists of two components:

Total Loss = Reconstruction Loss + KL Divergence Loss

#### Reconstruction Loss

Measures the difference between the input image and its reconstruction.

This experiment uses Binary Cross-Entropy (BCE).

#### KL Divergence Loss

Encourages the learned latent distribution to remain close to a standard normal distribution.

KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

## MNIST Dataset

MNIST contains 70,000 grayscale handwritten digit images.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28
- Classes: Digits 0–9
- Pixel values: 0–255

Pixel values are normalized to the range 0–1.

## Architecture

```text
           MNIST Image
             28 × 28
                |
                v
           Flatten Layer
           784 Features
                |
                v
         Dense Layer (256)
                |
          +-----+-----+
          |           |
          v           v
       z_mean     z_log_var
          |           |
          +-----+-----+
                |
                v
       Reparameterization
                |
                v
          Latent Vector
          2 Dimensions
                |
                v
         Dense Layer (256)
                |
                v
       Dense Layer (784)
                |
                v
          Reshape Layer
                |
                v
        Reconstructed Image
             28 × 28
```

## Algorithm

Step 1: Import the required libraries.

Step 2: Load the MNIST dataset.

Step 3: Normalize pixel values between 0 and 1.

Step 4: Build the Encoder network.

Step 5: Calculate latent mean and log-variance.

Step 6: Implement the reparameterization trick.

Step 7: Build the Decoder network.

Step 8: Define reconstruction and KL divergence losses.

Step 9: Train the VAE using a custom training step.

Step 10: Evaluate reconstruction performance.

Step 11: Reconstruct test images.

Step 12: Generate new digit images from random latent vectors.

Step 13: Plot training and validation losses.

Step 14: Save the Encoder, Decoder, and generated images.

## Software Requirements

- Python 3.10 or 3.11
- TensorFlow
- NumPy
- Matplotlib
- VS Code / Jupyter Notebook / Google Colab

## Installation

```bash
python -m pip install -r requirements.txt
```

## Execution

```bash
python vae.py
```

## Expected Output

The program displays:

- MNIST dataset shapes
- Encoder and Decoder model summaries
- Training reconstruction loss
- Training KL divergence loss
- Validation loss
- Final evaluation results

Generated files:

```text
outputs/reconstructed_images.png
outputs/generated_digits.png
outputs/training_loss.png

models/vae_encoder.keras
models/vae_decoder.keras
```

## Applications

1. Handwritten digit generation
2. Image synthesis
3. Representation learning
4. Anomaly detection
5. Data augmentation
6. Image reconstruction
7. Probabilistic generative modeling

## Result

A Variational Autoencoder was implemented and trained on the MNIST dataset using reconstruction loss and KL divergence.

The trained model can reconstruct handwritten digits and generate new digit-like images from latent samples.

## Conclusion

This experiment demonstrates how Variational Autoencoders learn probabilistic latent representations and use them to generate new data.

VAEs provide a foundation for advanced probabilistic generative modeling.
