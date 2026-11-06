#This program implements a two-component Gaussian Mixture Model from scratch, using NumPy rather than a prebuilt GMM library.

import numpy as np
import matplotlib.pyplot as plt

# Experiment 2: Gaussian Mixture Model
# Implementation using Expectation-Maximization

np.random.seed(42)

# Step 1: Generate synthetic data
data1 = np.random.normal(-2, 1, 500)
data2 = np.random.normal(3, 1.5, 500)

data = np.concatenate([data1, data2])

# Step 2: Initialize parameters
K = 2

means = np.array([-1.0, 2.0])
variances = np.array([1.0, 1.0])
weights = np.array([0.5, 0.5])


# Gaussian probability density function
def gaussian(x, mean, variance):
    return (
        1 / np.sqrt(2 * np.pi * variance)
    ) * np.exp(
        -((x - mean) ** 2) / (2 * variance)
    )


# Step 3: Expectation-Maximization
iterations = 100

for i in range(iterations):

    # E-Step: Calculate responsibilities
    responsibilities = np.zeros((len(data), K))

    for k in range(K):
        responsibilities[:, k] = (
            weights[k] *
            gaussian(data, means[k], variances[k])
        )

    total = responsibilities.sum(axis=1, keepdims=True)

    responsibilities = responsibilities / np.maximum(
        total, 1e-300
    )

    # M-Step: Update parameters
    Nk = responsibilities.sum(axis=0)

    for k in range(K):

        means[k] = np.sum(
            responsibilities[:, k] * data
        ) / Nk[k]

        variances[k] = np.sum(
            responsibilities[:, k] *
            (data - means[k]) ** 2
        ) / Nk[k]

        variances[k] = max(variances[k], 1e-6)

        weights[k] = Nk[k] / len(data)


# Step 4: Display results
print("GENERATIVE AI LAB - EXPERIMENT 2")
print("Gaussian Mixture Model")

print("\nEstimated Means:", means)
print("Estimated Variances:", variances)
print("Mixture Weights:", weights)

# Step 5: Calculate final log-likelihood
likelihood = np.zeros(len(data))

for k in range(K):
    likelihood += (
        weights[k] *
        gaussian(data, means[k], variances[k])
    )

log_likelihood = np.sum(
    np.log(np.maximum(likelihood, 1e-300))
)

print("Final Log-Likelihood:", round(log_likelihood, 4))

# Step 6: Plot data
x = np.linspace(-7, 9, 1000)

plt.figure(figsize=(10, 6))

plt.hist(
    data,
    bins=40,
    density=True,
    alpha=0.5,
    color="skyblue",
    edgecolor="white",
    label="Generated Data"
)

# Step 7: Plot Gaussian components
mixture_pdf = np.zeros_like(x)

for k in range(K):

    component_pdf = (
        weights[k] *
        gaussian(x, means[k], variances[k])
    )

    mixture_pdf += component_pdf

    plt.plot(
        x,
        component_pdf,
        linestyle="--",
        linewidth=2,
        label=f"Gaussian Component {k + 1}"
    )

# Step 8: Plot combined GMM
plt.plot(
    x,
    mixture_pdf,
    color="red",
    linewidth=3,
    label="Combined GMM"
)

plt.title("Gaussian Mixture Model from Scratch")
plt.xlabel("Data Values")
plt.ylabel("Probability Density")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "gmm_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved as gmm_distribution.png")
print("Experiment 2 completed successfully!")
