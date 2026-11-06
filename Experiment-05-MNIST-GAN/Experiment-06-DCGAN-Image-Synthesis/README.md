
# Experiment 6: DCGAN for Image Synthesis

## Aim

To implement a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow/Keras and train it on the MNIST dataset to generate synthetic handwritten digit images.

## Objectives

1. Understand the architecture of a DCGAN.
2. Learn the difference between GAN and DCGAN.
3. Load and preprocess the MNIST dataset.
4. Implement a convolutional Generator.
5. Implement a convolutional Discriminator.
6. Train the networks using adversarial learning.
7. Generate synthetic handwritten digit images.
8. Visualize generated images at different epochs.
9. Plot Generator and Discriminator losses.

## Theory

### What is a DCGAN?

A Deep Convolutional Generative Adversarial Network (DCGAN) is a type of GAN that uses convolutional neural networks for image generation.

A DCGAN contains two neural networks:

1. Generator
2. Discriminator

The Generator creates synthetic images from random noise.

The Discriminator distinguishes real images from generated images.

Both networks learn through adversarial training.

### 1. Generator

The Generator transforms a random noise vector into an image.

It uses:

- Dense layers
- Reshape layers
- Conv2DTranspose layers
- Batch Normalization
- ReLU activation
- Tanh output activation

Conv2DTranspose layers increase the spatial dimensions of feature maps.

### 2. Discriminator

The Discriminator classifies images as real or fake.

It uses:

- Conv2D layers
- LeakyReLU activation
- Dropout
- Flatten
- Dense output layer

Convolutional layers learn spatial features such as edges, shapes, and patterns.

## DCGAN Architecture

```text
            Random Noise
           (100 Features)
                 |
                 v
             GENERATOR
                 |
                 v
          Dense + Reshape
            (7 × 7 × 256)
                 |
                 v
          Conv2DTranspose
           (14 × 14 × 128)
                 |
                 v
          Conv2DTranspose
            (28 × 28 × 64)
                 |
                 v
          Conv2DTranspose
             (28 × 28 × 1)
                 |
                 v
          Generated Image
                 |
                 v
            DISCRIMINATOR
                 |
                 v
          Convolution Layers
                 |
                 v
           Flatten + Dense
                 |
                 v
           Real / Fake Score
```

Real MNIST images are also supplied to the Discriminator during training.

## Mathematical Objective

The original GAN minimax objective is:

min_G max_D V(D,G)

V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:

- G = Generator
- D = Discriminator
- x = Real image
- z = Random noise vector
- G(z) = Generated image
- D(x) = Probability that a real image is classified as real

This experiment uses binary cross-entropy with a non-saturating Generator objective.

Generator Loss:

L_G = BCE(1, D(G(z)))

Discriminator Loss:

L_D = BCE(1, D(x)) + BCE(0, D(G(z)))

The Discriminator outputs logits, so binary cross-entropy is calculated using from_logits=True.

## MNIST Dataset

The MNIST dataset contains 70,000 handwritten digit images.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28
- Channels: 1 (grayscale)
- Classes: Digits 0–9

This experiment uses the training images.

Pixel values are normalized to [-1, 1].

## Generator Architecture

Input:
100-dimensional random noise vector

Layers:

1. Dense: 7 × 7 × 256
2. BatchNormalization
3. LeakyReLU
4. Reshape: 7 × 7 × 256
5. Conv2DTranspose: 128 filters, stride 1
6. BatchNormalization
7. LeakyReLU
8. Conv2DTranspose: 64 filters, stride 2
9. BatchNormalization
10. LeakyReLU
11. Conv2DTranspose: 1 filter, stride 2, Tanh

Output:
28 × 28 × 1 image

## Discriminator Architecture

Input:
28 × 28 × 1 image

Layers:

1. Conv2D: 64 filters, stride 2
2. LeakyReLU
3. Dropout
4. Conv2D: 128 filters, stride 2
5. LeakyReLU
6. Dropout
7. Flatten
8. Dense: 1 logit

Output:
Real/Fake classification score

## Training Configuration

- Epochs: 20
- Batch Size: 256
- Noise Dimension: 100
- Optimizer: Adam
- Learning Rate: 0.0002
- Beta 1: 0.5
- Loss: Binary Cross-Entropy

## Algorithm

Step 1: Import TensorFlow, NumPy, and Matplotlib.

Step 2: Load the MNIST training dataset.

Step 3: Normalize images to [-1, 1].

Step 4: Create a TensorFlow Dataset.

Step 5: Build the convolutional Generator.

Step 6: Build the convolutional Discriminator.

Step 7: Define Generator and Discriminator losses.

Step 8: Initialize separate Adam optimizers.

Step 9: Generate synthetic images using random noise.

Step 10: Calculate Discriminator predictions for real and fake images.

Step 11: Calculate adversarial losses.

Step 12: Update both networks using gradients.

Step 13: Repeat training for 20 epochs.

Step 14: Save generated images at selected epochs.

Step 15: Plot Generator and Discriminator training losses.

Step 16: Save the trained models.

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
python dcgan.py
```

## Expected Output

The program displays:

- MNIST dataset shape
- Generator model summary
- Discriminator model summary
- Generator loss per epoch
- Discriminator loss per epoch
- Generated image file paths
- Saved model file paths

Generated files:

```text
outputs/generated_epoch_01.png
outputs/generated_epoch_05.png
outputs/generated_epoch_10.png
outputs/generated_epoch_20.png
outputs/generated_digits.png
outputs/training_loss.png

models/dcgan_generator.keras
models/dcgan_discriminator.keras
```

## Observations

At early epochs, the Generator may produce noisy or unclear images.

As adversarial training progresses, the images may become more recognizable.

GAN training is not guaranteed to improve monotonically.

Generator and Discriminator losses may fluctuate.

Image quality should be assessed visually rather than using loss values alone.

## Applications

1. Synthetic image generation
2. Image synthesis
3. Data augmentation
4. Generative AI research
5. Image-to-image modeling foundations
6. Creative AI applications
7. Medical image synthesis research

## Result

A DCGAN architecture was implemented using convolutional Generator and Discriminator networks for MNIST image synthesis.

The implementation trains the networks adversarially and saves generated images, training graphs, and trained models.

## Conclusion

This experiment demonstrates how convolutional layers improve a GAN's ability to learn spatial image patterns.

DCGANs provide an important foundation for more advanced deep generative image models.
