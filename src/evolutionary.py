# Implements the evolutionary algorithm, including fitness, mutation, and crossover
import numpy as np

# Simple mutation: randomly adjust features
def mutate(features, mutation_rate=0.1):
    mutated = features.copy()
    for i in range(mutated.shape[0]):
        if np.random.rand() < mutation_rate: # DevSkim: ignore DS148264
            idx = np.random.randint(mutated.shape[1])
            mutated[i, idx] += np.random.choice([-1, 1])
    return mutated

# Simple crossover: swap parts between two feature sets
def crossover(parent1, parent2):
    point = np.random.randint(1, parent1.shape[0] - 1)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2
