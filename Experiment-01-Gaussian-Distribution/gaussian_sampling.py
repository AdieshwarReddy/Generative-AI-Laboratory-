
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mean = 0
std_dev = 1
num_samples = 10000

# Generate Gaussian samples
np.random.seed(42)

samples = np.random.normal(
    mean,
    std_dev,
    num_samples
)

# Calculate statistics
print("GENERATIVE AI LAB - EXPERIMENT 1")
print("Gaussian Distribution Sampling")

print("Mean:", np.mean(samples))
print("Standard Deviation:", np.std(samples))

# Plot histogram
plt.figure(figsize=(10, 6))

plt.hist(
    samples,
    bins=50,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Generated Samples"
)

# Theoretical Gaussian curve
x = np.linspace(-4, 4, 1000)

pdf = (
    1 / (std_dev * np.sqrt(2 * np.pi))
) * np.exp(
    -0.5 * ((x - mean) / std_dev) ** 2
)

plt.plot(
    x,
    pdf,
    color="red",
    linewidth=2,
    label="Gaussian PDF"
)

plt.title("Gaussian Distribution Sampling")
plt.xlabel("Sample Values")
plt.ylabel("Probability Density")

plt.legend()
plt.grid(alpha=0.3)

# Save graph
plt.savefig(
    "gaussian_distribution.png",
    dpi=300
)

plt.show()

print("Graph generated successfully!")
