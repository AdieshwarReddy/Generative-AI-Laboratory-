
# Experiment 1: Gaussian Distribution Sampling

## Aim
To implement a simple probabilistic model by generating random samples from a Gaussian (Normal) distribution using Python.

## Objectives
- Understand Gaussian distribution.
- Generate random samples using NumPy.
- Calculate mean and standard deviation.
- Visualize the Gaussian distribution using Matplotlib.
- Understand probabilistic sampling in Generative AI.

## Theory
A Gaussian distribution, also known as a Normal distribution, is a continuous probability distribution characterized by a bell-shaped curve.

It is defined by:
- Mean (μ): Center of the distribution.
- Standard Deviation (σ): Spread of the distribution.

### Gaussian Probability Density Function

f(x) = (1 / (σ√(2π))) × e^(-(x-μ)² / (2σ²))

Gaussian distributions are widely used in machine learning, statistical modeling, VAEs, and diffusion models.

## Algorithm
1. Import NumPy and Matplotlib.
2. Define mean, standard deviation, and sample size.
3. Generate random samples using a Gaussian distribution.
4. Calculate sample mean and standard deviation.
5. Plot a histogram of the generated samples.
6. Plot the theoretical Gaussian probability density curve.
7. Display and save the graph.

## Software Requirements
- Python 3
- NumPy
- Matplotlib
- VS Code / Jupyter Notebook / Google Colab

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python gaussian_sampling.py
```

## Expected Output
- Gaussian random samples generated.
- Sample mean approximately equal to 0.
- Sample standard deviation approximately equal to 1.
- Bell-shaped Gaussian distribution graph generated.
- Graph saved as gaussian_distribution.png.

## Applications
- Generative AI
- Variational Autoencoders
- Diffusion Models
- Statistical Modeling
- Noise Generation

## Result
Successfully implemented Gaussian distribution sampling using Python and visualized the probability distribution.

## Conclusion
This experiment demonstrates probabilistic sampling and provides a foundation for understanding generative models.
