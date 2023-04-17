import torch
import numpy as np


def add_noise_laplace(matrix, sensitivity, epsilon):
    # Convert the matrix to a PyTorch tensor
    #matrix = torch.tensor(matrix, dtype=torch.float32)

    # Compute the scale parameter for the Laplace distribution
    b = sensitivity / epsilon

    # Generate noise from the Laplace distribution with scale b
    noise = torch.tensor(np.random.laplace(scale=b, size=matrix.shape), dtype=torch.float32)

    # Add the noise to the matrix
    noisy_matrix = matrix + noise

    return noisy_matrix



matrix = [[1,2,3,4],[5,6,7,8]]
matrix = torch.tensor(matrix, dtype=torch.float32)
sensitivity = 1
epsilon = 0.1

noisy_matrix = add_noise_laplace(matrix, sensitivity, epsilon)

print(noisy_matrix)


def add_noise(data, epsilon, sensitivity):
    noise_tesnor = np.random.laplace(1, sensitivity / epsilon, data.shape)
    # data = torch.add(data, torch.from_numpy(noise_tesnor))
    # for x in np.nditer(np_data, op_flags=['readwrite']):
    #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
    return data.add(torch.from_numpy(noise_tesnor).float())




noisy_matrix = add_noise(matrix, 1, 1)

print(noisy_matrix)

import torch


def laplace_mechanism(matrix, epsilon, sensitivity=1.0):
    """
    Apply the Laplace mechanism for differential privacy.

    Args:
        matrix (torch.Tensor): The input matrix.
        epsilon (float): Privacy parameter. Lower values provide more privacy.
        sensitivity (float): The sensitivity of the query. Default is 1.0.

    Returns:
        torch.Tensor: The matrix with Laplace noise added.
    """
    scale = sensitivity / epsilon
    noise = torch.tensor(torch.rand_like(matrix).numpy() - 0.5) * 2  # Uniform distribution in the range [-1, 1]
    noise = torch.clamp(noise, min=-0.49 + 1e-10, max=0.49 - 1e-10)  # Clip the noise to avoid edge cases
    laplace_noise = -scale * torch.sign(noise) * torch.log(1 - 2 * torch.abs(noise))
    return matrix + laplace_noise


matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
epsilon = 1.0  # Adjust this value to control the level of privacy

noisy_matrix = laplace_mechanism(matrix, epsilon)
print(noisy_matrix)

import torch
import numpy as np


def laplace_mechanism(matrix, epsilon, sensitivity=1.0):
    """
    Apply the Laplace mechanism for differential privacy.

    Args:
        matrix (torch.Tensor): The input matrix.
        epsilon (float): Privacy parameter. Lower values provide more privacy.
        sensitivity (float): The sensitivity of the query. Default is 1.0.

    Returns:
        torch.Tensor: The matrix with Laplace noise added.
    """
    scale = sensitivity / epsilon
    laplace_noise = torch.tensor(np.random.laplace(0, scale, matrix.shape), dtype=torch.float32)
    return matrix + laplace_noise


matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
epsilon = 1.0  # Adjust this value to control the level of privacy

noisy_matrix = laplace_mechanism(matrix, epsilon)
print(noisy_matrix)
