
# Experiment 5: Generative Adversarial Network (GAN) on MNIST

## Aim

To implement a basic Generative Adversarial Network (GAN) consisting of a Generator and a Discriminator using TensorFlow/Keras and train it on MNIST handwritten digit images.

## Objectives

1. Understand the architecture of Generative Adversarial Networks.
2. Learn the roles of the Generator and Discriminator.
3. Load and preprocess the MNIST dataset.
4. Implement a basic fully connected GAN.
5. Train both networks using adversarial learning.
6. Generate synthetic handwritten digit images.
7. Visualize the generated images at different training epochs.
8. Plot Generator and Discriminator training losses.

## Theory

### What is a GAN?

A Generative Adversarial Network (GAN) is a generative deep learning model introduced by Ian Goodfellow and colleagues in 2014.

A GAN consists of two neural networks:

1. Generator (G)
2. Discriminator (D)

These networks compete during training.

The Generator attempts to create realistic synthetic images.

The Discriminator attempts to distinguish real images from generated images.

Through adversarial training, the Generator gradually learns to produce images resembling the training dataset.

## 1. Generator

The Generator receives a random noise vector as input and transforms it into a synthetic image.

Input:
Random Noise Vector (100 dimensions)

Output:
Generated MNIST Image (28 × 28 pixels)

The Generator learns to produce images that can fool the Discriminator.

## 2. Discriminator

The Discriminator receives an image and predicts whether it is real or generated.

Input:
MNIST Image (28 × 28 pixels)

Output:
Probability between 0 and 1

A value close to 1 indicates that the image is predicted to be real.

A value close to 0 indicates that the image is predicted to be fake.

## GAN Architecture

```text
        Random Noise Vector
           (100 Features)
                 |
                 v
             GENERATOR
                 |
                 v
         Fake MNIST Image
             (28 × 28)
                 |
                 v
           DISCRIMINATOR <----- Real MNIST Image
                 |
                 v
          Real / Fake Score
                 |
                 v
          Adversarial Loss
                 |
                 v
        Update Neural Networks
```

## Mathematical Objective

The original GAN minimax objective is:

min_G max_D V(D,G)

V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:

- G = Generator
- D = Discriminator
- x = Real training image
- z = Random noise vector
- G(z) = Generated image
- D(x) = Discriminator's estimate that a real image is real
- D(G(z)) = Discriminator's estimate that a generated image is real

In this experiment, the Generator uses the common non-saturating loss:

Generator Loss = -E[log D(G(z))]

The Discriminator minimizes binary cross-entropy on real and generated images.

## MNIST Dataset

The MNIST dataset contains 70,000 grayscale handwritten digit images.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28
- Classes: 0 to 9
- Pixel range: 0 to 255

This experiment uses the 60,000 training images.

Images are normalized to the range [-1, 1].

## Software Requirements

- Python 3.10 or 3.11
- TensorFlow
- NumPy
- Matplotlib
- VS Code / Jupyter Notebook / Google Colab

## Algorithm

Step 1: Import TensorFlow, NumPy, and Matplotlib.

Step 2: Load the MNIST training dataset.

Step 3: Normalize images to [-1, 1].

Step 4: Create the Generator neural network.

Step 5: Create the Discriminator neural network.

Step 6: Define Binary Cross-Entropy loss.

Step 7: Create separate Adam optimizers.

Step 8: Generate fake images using random noise.

Step 9: Train the Discriminator using real and fake images.

Step 10: Train the Generator to fool the Discriminator.

Step 11: Repeat adversarial training for 20 epochs.

Step 12: Save generated images at selected epochs.

Step 13: Plot Generator and Discriminator losses.

Step 14: Save the trained Generator and Discriminator.

## Generator Architecture

Input Noise: 100 features

Dense Layer: 256 neurons, ReLU

Dense Layer: 512 neurons, ReLU

Dense Layer: 1024 neurons, ReLU

Output Layer: 784 neurons, Tanh

Reshape: 28 × 28 × 1

## Discriminator Architecture

Input Image: 28 × 28 × 1

Flatten: 784 features

Dense Layer: 512 neurons, LeakyReLU

Dense Layer: 256 neurons, LeakyReLU

Output Layer: 1 neuron, Sigmoid

## Training Configuration

- Epochs: 20
- Batch Size: 256
- Noise Dimension: 100
- Optimizer: Adam
- Learning Rate: 0.0002
- Beta 1: 0.5
- Loss: Binary Cross-Entropy

## Installation

```bash
python -m pip install -r requirements.txt
```

## Execution

```bash
python gan.py
```

## Expected Output

The program displays:

- Dataset shape
- Generator model summary
- Discriminator model summary
- Generator loss for each epoch
- Discriminator loss for each epoch
- Saved generated images
- Saved training loss graph

Generated files:

```text
outputs/generated_epoch_01.png
outputs/generated_epoch_05.png
outputs/generated_epoch_10.png
outputs/generated_epoch_20.png
outputs/generated_digits.png
outputs/training_loss.png

models/gan_generator.keras
models/gan_discriminator.keras
```

## Observations

At the beginning of training, the Generator usually produces noisy images.

As training progresses, the generated images may gradually resemble handwritten digits.

GAN training is not guaranteed to improve monotonically and may be unstable.

The Generator and Discriminator losses can fluctuate during training.

## Applications

1. Synthetic image generation
2. Data augmentation
3. Image synthesis
4. Representation learning
5. Creative AI
6. Medical image research
7. Generative modeling

## Result

A basic Generative Adversarial Network was implemented using TensorFlow/Keras.

The Generator and Discriminator were designed for adversarial training on the MNIST dataset, and the program generates synthetic handwritten digit images.

## Conclusion

This experiment demonstrates the working principle of Generative Adversarial Networks.

The Generator learns to produce synthetic images while the Discriminator learns to distinguish real images from generated images.

GANs form an important foundation for advanced image generation models such as DCGANs.
