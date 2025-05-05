# Implements the evolutionary algorithm, including fitness, mutation, and crossover
import numpy as np

# Simple mutation: randomly adjust features
# random, simple with small perturbations (e.g., +/-1)
# (perturbation: small change or deviation from the original value)
def mutate(features, mutation_rate=0.1):
    mutated = features.copy()
    for i in range(mutated.shape[0]):
        # np.random.rand() generates a random float in the range [0.0, 1.0)
        # if the random number is less than mutation_rate, mutate the feature
        if np.random.rand() < mutation_rate: # DevSkim: ignore DS148264
            idx = np.random.randint(mutated.shape[1])
            mutated[i, idx] += np.random.choice([-1, 1])
    return mutated

# Simple crossover: swap parts between two feature vectors (per row basis)
def crossover(parent1, parent2):
    parent1 = parent1.copy()
    parent2 = parent2.copy()
    
    # Ensure both parents are 1D array by squeezing the first dimension if necessary
    if parent1.ndim == 2:
        parent1 = parent1.squeeze(0)
    if parent2.ndim == 2:
        parent2 = parent2.squeeze(0)

    # Randomly select a crossover point
    point = np.random.randint(1, parent1.shape[0] - 1)
    # Create two children by swapping parts of the parents
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2
