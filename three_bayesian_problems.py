# Bayesian Analysis of Study Hours for Students
# This code simulates a Bayesian model to estimate the average number of study hours per day for students.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

bold_start = "\033[1m"
bold_end = "\033[0m"

print(f"{bold_start}Problem 1:{bold_end}\nThis Bayesian model estimates the average number of study hours per day for students. We start with a prior belief that the average is 4 hours, and update this belief after collecting data from 100 students.")

# Step 1: Prior beliefs about the population
prior_mu = 4 # Prior belief about average study hours (mean)
prior_precision = 1 # Precision = 1/variance (small precision = uncertain prior)

prior_sigma_alpha = 2 # Prior shape for variance
prior_sigma_beta = 2 # Prior scale for variance (used for gamma distribution)

# Step 2: Simulate observed data (real-world sample)
np.random.seed(42) # Ensures reproducibility
true_mu = 5 # True unknown average study hours
true_sigma = 1.5 # True unknown standard deviation

n = 100 # Sample size = number of students
data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)  # Sample data from true distribution

# Step 3: Update posterior parameters for mean (using Normal distribution)
posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision

# Step 4: Update posterior parameters for variance (using Gamma distribution)
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

# Step 5: Sample from posterior distributions
posterior_mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
posterior_sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Step 6: Plotting posterior distributions
plt.figure(figsize=(10, 4))

# Posterior for mean study hours
plt.subplot(1, 2, 1)
plt.hist(posterior_mu_samples, bins=30, density=True, color='orange', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\mu$ (Study Hours)')
plt.xlabel('Study Hours')
plt.ylabel('Density')

# Posterior for standard deviation
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma_samples, bins=30, density=True, color='teal', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\sigma$')
plt.xlabel('Standard Deviation')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Step 7: Summary statistics
print("Mean of μ (study hours):", np.mean(posterior_mu_samples))
print("Std Dev of μ:", np.std(posterior_mu_samples))
print("Mean of σ:", np.mean(posterior_sigma_samples))
print("Std Dev of σ:", np.std(posterior_sigma_samples))


# Bayesian Estimation of Fresh Graduate Monthly Salaries
# This model uses Bayesian inference to estimate the true average monthly salary and its uncertainty among fresh graduates.
print(f"\n\n{bold_start}Problem 2:{bold_end}\nThe goal is to estimate the average monthly salary of fresh graduates using Bayesian inference. We begin with a prior assumption (₱20,000) and refine it using data from 80 graduates.")

# Prior beliefs
prior_mu = 20000 # Prior belief of average salary
prior_precision = 1 / 5000**2  # Variance = 5000^2 => Precision = 1/variance

prior_sigma_alpha = 3
prior_sigma_beta = 2e7# Chosen to reflect moderate uncertainty

# Simulated observed data
np.random.seed(7)
true_mu = 22000
true_sigma = 4000
n = 80

data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

# Posterior for μ
posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision

# Posterior for σ
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = posterior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Sampling
mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(mu_samples, bins=30, density=True, color='goldenrod', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\mu$ (Salary in ₱)')
plt.xlabel('Monthly Salary in ₱')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(sigma_samples, bins=30, density=True, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\sigma$')
plt.xlabel('Salary SD in ₱')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

print("Estimated Mean Salary:", np.mean(mu_samples))
print("Estimated Std Dev of Mean:", np.std(mu_samples))
print("Estimated Salary SD:", np.mean(sigma_samples))
print("SD of Salary SD:", np.std(sigma_samples))


# Bayesian Estimation of Average Height of College Students
# This analysis updates prior beliefs about the average height of college students using a Bayesian framework.
print(f"\n\n{bold_start}Problem 3:{bold_end}\nEstimate the true average height of college students using Bayesian inference. We begin with a prior belief that students are ~165 cm tall, and refine this based on measurements from 60 students.")

# Prior belief
prior_mu = 165
prior_precision = 1 / 15**2 # SD = 15cm → low precision

prior_sigma_alpha = 2
prior_sigma_beta = 100 # Loose belief for variance

# Simulated data
np.random.seed(21)
true_mu = 168
true_sigma = 10
n = 60
data = np.random.normal(loc=true_mu, scale=true_sigma, size=n)

# Posterior for μ
posterior_precision = prior_precision + n / true_sigma**2
posterior_mu = (prior_precision * prior_mu + np.sum(data) / true_sigma**2) / posterior_precision

# Posterior for σ
posterior_sigma_alpha = prior_sigma_alpha + n / 2
posterior_sigma_beta = posterior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Samples
mu_samples = np.random.normal(posterior_mu, 1 / np.sqrt(posterior_precision), size=10000)
sigma_samples = np.sqrt(np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(mu_samples, bins=30, density=True, color='orchid', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\mu$ (Height)')
plt.xlabel('Height in cm')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(sigma_samples, bins=30, density=True, color='slateblue', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of $\sigma$')
plt.xlabel('Height SD in cm')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

print("Estimated Mean Height:", np.mean(mu_samples))
print("Std Dev of Mean Height:", np.std(mu_samples))
print("Estimated Height SD:", np.mean(sigma_samples))
print("SD of Height SD:", np.std(sigma_samples))
