### The PRIOR

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.63, 1.8, num=50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1

# sneaky advanced note: I am using the uniform distribution for clarity, 
# but we can also make the beta distribution look completely flat by tweaking alpha and beta!

uniform_dist = uniform_dist / uniform_dist.sum()
beta_dist = sts.beta.pdf(mu, a=2, b=2)  # Example beta distribution
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

### THE LIKELIHOOD

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)  # note that mu here is an array of values, so the output is also an array!
    return likelihood_out / sts.norm.pdf(1.7, mu, scale=0.1)

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()

### POSTERIOR 

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()