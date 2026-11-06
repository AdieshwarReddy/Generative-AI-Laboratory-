
# Experiment 2: Gaussian Mixture Model from Scratch

## Aim

To implement a Gaussian Mixture Model (GMM) using Python and understand how multiple Gaussian distributions can be combined to represent complex data distributions.

## Objectives

- Understand Gaussian Mixture Models.
- Generate synthetic data from multiple Gaussian distributions.
- Implement the Expectation-Maximization (EM) algorithm.
- Estimate the means, variances, and mixture weights.
- Visualize the generated data and learned Gaussian components.
- Understand the role of GMMs in Generative AI.

## Theory

### What is a Gaussian Mixture Model?

A Gaussian Mixture Model (GMM) is a probabilistic model that represents a dataset as a weighted combination of multiple Gaussian distributions.

Unlike a single Gaussian distribution, a GMM can represent data containing multiple clusters.

### Mathematical Formula

P(x) = Σ [πₖ × N(x | μₖ, σₖ²)]

Where:

- K = Number of Gaussian components
- πₖ = Mixture weight of component k
- μₖ = Mean of component k
- σₖ² = Variance of component k
- N = Gaussian probability density function

The mixture weights satisfy:

Σπₖ = 1

### Expectation-Maximization Algorithm

The EM algorithm estimates the parameters of a Gaussian Mixture Model.

**1. Initialization**

Initialize means, variances, and mixture weights.

**2. Expectation Step (E-Step)**

Calculate the probability that each data point belongs to each Gaussian component.

These probabilities are called responsibilities.

**3. Maximization Step (M-Step)**

Update the parameters using the calculated responsibilities:

- Means
- Variances
- Mixture weights

**4. Repeat**

Repeat the E-Step and M-Step until convergence or until the maximum number of iterations is reached.

## Flowchart

```text
           START
             |
             v
     Import Libraries
             |
             v
    Generate Sample Data
             |
             v
    Initialize Parameters
             |
             v
       E-Step
 Calculate Responsibilities
             |
             v
       M-Step
 Update Model Parameters
             |
             v
    Check Convergence
         /       \
       No         Yes
       |           |
       +--> E-Step |
                   v
         Visualize Results
                   |
                   v
                  END
```

## Algorithm

1. Import NumPy and Matplotlib.
2. Generate synthetic data from two Gaussian distributions.
3. Initialize component means, variances, and weights.
4. Calculate responsibilities using the E-Step.
5. Update model parameters using the M-Step.
6. Repeat for a fixed number of iterations.
7. Calculate the probability density of each component.
8. Plot the histogram and learned GMM density.
9. Display the estimated parameters.

## Software Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- VS Code / Jupyter Notebook / Google Colab

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python gmm.py
```

## Expected Output

The program displays:

- Estimated Gaussian means
- Estimated Gaussian variances
- Estimated mixture weights
- Final log-likelihood
- A histogram of generated data
- Learned Gaussian component curves
- Combined GMM probability density curve

The graph is saved as:

gmm_distribution.png

## Applications

1. Generative AI
2. Data clustering
3. Anomaly detection
4. Image segmentation
5. Speech recognition
6. Probability density estimation

## Result

A Gaussian Mixture Model was implemented using the Expectation-Maximization algorithm to estimate the parameters of a mixture of Gaussian distributions.

## Conclusion

This experiment demonstrates how multiple Gaussian distributions can represent complex data.

Gaussian Mixture Models are important probabilistic generative models used in machine learning and data analysis.
