# Implements the evolutionary algorithm, including fitness, mutation, and crossover
import numpy as np

# Mutation for segment indices
def mutate_segment(segment, mutation_rate=0.1, signal_length=None):
    start, end = segment
    if np.random.rand() < mutation_rate: # DevSkim: ignore DS148264
        start = max(0, start + np.random.choice([-1, 1]))
    if np.random.rand() < mutation_rate: # DevSkim: ignore DS148264
        end = min(signal_length - 1, end + np.random.choice([-1, 1]))
    return (start, end)

# Crossover for segment indices
def crossover_segments(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2
    # Swap start and end points
    return (start1, end2), (start2, end1)

# Improved fitness function: evaluate based on pattern matching
def evaluate_fitness(segment, signal):
    start, end = segment
    segment_data = signal[start:end]
    if len(segment_data) < 5:
        return -np.inf  # Penalize too-short segments

    # --- Feature-based metrics ---
    mean = np.mean(segment_data)
    std = np.std(segment_data)
    max_val = np.max(segment_data)
    min_val = np.min(segment_data)
    # Entropy (discretize to 10 bins)
    hist, _ = np.histogram(segment_data, bins=10, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0

    # --- Correlation with R-wave shape (simple template) ---
    # Use a synthetic R-wave template (sharp positive peak)
    template = np.zeros_like(segment_data)
    if len(template) > 0:
        template[len(template)//2] = 1.0
        corr = np.corrcoef(segment_data, template)[0, 1]
    else:
        corr = 0

    # --- Combine metrics ---
    # Encourage high max (peak), high std, high entropy, and R-wave-like shape
    score = 1.0 * max_val + 0.5 * std + 0.2 * entropy + 0.5 * abs(corr)
    return score

def run_evolutionary_segmentation(signal, generations=10, population_size=5):
    """
    Example evolutionary approach to learn segmentation indices.
    """
    # Initialize random population (each individual is a set of random segment borders)
    population = []
    for _ in range(population_size):
        start = np.random.randint(0, len(signal)//2)
        end = np.random.randint(len(signal)//2, len(signal))
        population.append((start, end))

    for gen in range(generations):
        # Evaluate fitness using the improved function
        fitness = [evaluate_fitness(ind, signal) for ind in population]
        # Select top half
        selected = [x for _, x in sorted(zip(fitness, population), reverse=True)][: population_size // 2]

        # Breed new individuals
        offspring = []
        while len(offspring) < population_size - len(selected):
            # Fix: np.random.choice expects a 1D array, so use indices
            idxs = np.random.choice(len(selected), 2, replace=True)
            p1, p2 = selected[idxs[0]], selected[idxs[1]]
            c1, c2 = crossover_segments(p1, p2)
            offspring.append(c1)
            if len(offspring) < population_size - len(selected):
                offspring.append(c2)

        # Mutate
        mutated = [mutate_segment(ind, mutation_rate=0.5, signal_length=len(signal)) for ind in offspring]

        # New population
        population = selected + mutated

    # Return best segmentation
    final_fitness = [evaluate_fitness(ind, signal) for ind in population]
    best_ind = population[np.argmax(final_fitness)]
    return best_ind
