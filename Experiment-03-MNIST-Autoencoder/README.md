
# Experiment 3: Autoencoder for MNIST Image Reconstruction

## Aim

To build and train a simple Autoencoder using the MNIST handwritten digit dataset and reconstruct input images using TensorFlow and Keras.

## Objectives

1. Understand the architecture of an Autoencoder.
2. Load and preprocess the MNIST dataset.
3. Implement an Encoder and Decoder neural network.
4. Train the Autoencoder to reconstruct handwritten digits.
5. Evaluate reconstruction performance.
6. Visualize original and reconstructed images.
7. Understand the importance of Autoencoders in Generative AI.

## Theory

### What is an Autoencoder?

An Autoencoder is an unsupervised/self-supervised neural network that learns to reconstruct its input data.

It consists of two main components:

1. Encoder
2. Decoder

### 1. Encoder

The Encoder converts the input data into a lower-dimensional representation called the latent representation or bottleneck.

For example:

Input Image: 784 features

Encoded Representation: 32 features

### 2. Decoder

The Decoder reconstructs the original image from the compressed latent representation.

For example:

Encoded Representation: 32 features

Reconstructed Image: 784 features

### Autoencoder Architecture

```text
       MNIST Input Image
            28 x 28
               |
               v
          Flatten Layer
          784 Features
               |
               v
          Dense Layer
          128 Neurons
               |
               v
         Encoder Layer
          32 Neurons
               |
               v
      Latent Representation
               |
               v
         Decoder Layer
          128 Neurons
               |
               v
         Output Layer
          784 Neurons
               |
               v
       Reconstructed Image
            28 x 28
```

### Mathematical Representation

Encoding:

z = f(Wx + b)

Decoding:

x' = g(W'z + b')

Where:

- x = Original input
- z = Latent representation
- x' = Reconstructed output
- W and W' = Weight matrices
- b and b' = Bias vectors
- f and g = Activation functions

### Reconstruction Loss

The Autoencoder minimizes the difference between the original and reconstructed images.

In this experiment, Mean Squared Error (MSE) is used.

MSE = (1/n) × Σ(x - x')²

A lower reconstruction loss indicates better reconstruction accuracy.

## MNIST Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits from 0 to 9.

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 x 28 pixels
- Number of Classes: 10
- Pixel Range: 0 to 255

The images are normalized to values between 0 and 1 before training.

## Software Requirements

- Python 3.10 or 3.11
- TensorFlow
- NumPy
- Matplotlib
- VS Code / Jupyter Notebook / Google Colab

## Algorithm

Step 1: Import the required Python libraries.

Step 2: Load the MNIST dataset using Keras.

Step 3: Normalize pixel values between 0 and 1.

Step 4: Flatten each 28 x 28 image into 784 features.

Step 5: Create the Encoder neural network.

Step 6: Create the Decoder neural network.

Step 7: Combine Encoder and Decoder into an Autoencoder.

Step 8: Compile the model using Adam optimizer and MSE loss.

Step 9: Train the model using MNIST training images.

Step 10: Evaluate the trained model on test images.

Step 11: Reconstruct test images using the trained model.

Step 12: Visualize original and reconstructed images.

Step 13: Save the trained model and generated graphs.

## Implementation

The complete Python implementation is available in:

`autoencoder.py`

## Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Execution

Run the program:

```bash
python autoencoder.py
```

## Expected Output

The program displays:

- Dataset shapes
- Autoencoder model summary
- Training and validation losses
- Final test reconstruction loss
- Reconstructed MNIST digit images

Generated files:

```text
outputs/reconstructed_images.png
outputs/training_loss.png
models/mnist_autoencoder.keras
```

## Applications

1. Image reconstruction
2. Image denoising
3. Dimensionality reduction
4. Feature extraction
5. Anomaly detection
6. Data compression research
7. Representation learning

## Result

A simple Autoencoder was implemented and trained on the MNIST dataset to reconstruct handwritten digit images.

## Conclusion

This experiment demonstrates how an Autoencoder learns compressed representations of images and reconstructs them using an Encoder-Decoder architecture.

Autoencoders provide a foundation for understanding advanced generative models such as Variational Autoencoders (VAEs).
