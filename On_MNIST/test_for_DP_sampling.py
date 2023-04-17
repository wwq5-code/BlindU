import numpy as np

def scoring_function(matrix):
    # This is a simple scoring function that returns the matrix itself as scores.
    # You can replace it with your own scoring function if needed.
    return matrix

def dp_sampling(matrix, epsilon, sample_size):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    # Flatten the matrix and probabilities for sampling
    flat_matrix = matrix.flatten()
    flat_probabilities = probabilities.flatten()

    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_matrix)),
        size=sample_size,
        replace=True,  # or False, when without replacement
        p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)

    # Set the sampled elements to their original values
    np.put(output_matrix, sampled_indices, flat_matrix[sampled_indices])

    return output_matrix

matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
epsilon = 1.0
sample_size = 4

sampled_matrix = dp_sampling(matrix, epsilon, sample_size)
print(sampled_matrix)


## a torch version

import torch

def scoring_function(matrix):
    # This is a simple scoring function that returns the matrix itself as scores.
    # You can replace it with your own scoring function if needed.
    return matrix

def dp_sampling(matrix, epsilon, sample_size):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = torch.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    print(probabilities)

    # Flatten the matrix and probabilities for sampling
    flat_matrix = matrix.flatten()
    flat_probabilities = probabilities.flatten()

    # Sample elements without replacement
    sampled_indices = torch.multinomial(flat_probabilities, sample_size, replacement=False)

    # Create the output matrix with 0s
    output_matrix = torch.zeros_like(matrix)

    # Set the sampled elements to their original values
    output_matrix.view(-1)[sampled_indices] = flat_matrix[sampled_indices]

    return output_matrix

matrix = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 0]], dtype=torch.float32)
epsilon = 1.0
sample_size = 3

sampled_matrix = dp_sampling(matrix, epsilon, sample_size)
print(sampled_matrix)
