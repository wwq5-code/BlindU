import numpy as np
import matplotlib.pyplot as plt

# Create a 3-channel image with random values
image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

# Display the image
# plt.imshow(image)
# plt.show()

#
# print(image)

import numpy as np

def scoring_function(matrix):
    # This is a simple scoring function that returns the mean value across channels as scores.
    # You can replace it with your own scoring function if needed.
    return matrix.mean(axis=1)

def dp_sampling(matrix, epsilon, sample_size):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    # Flatten the scores and probabilities for sampling
    flat_scores = scores.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)

    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_scores)),
        size=sample_size,
        replace=False,
        p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)

    # Set the sampled elements to their original values for all channels
    for channel in range(matrix.shape[1]):
        np.put(output_matrix[0, channel], sampled_indices, matrix[0, channel].flatten()[sampled_indices])

    return output_matrix

matrix = np.random.rand(1, 3, 3, 3)
epsilon = 1.0
sample_size = 8

sampled_matrix = dp_sampling(matrix, epsilon, sample_size)
print(sampled_matrix)




import numpy as np

def scoring_function(matrix):
    # This is a simple scoring function that returns the mean value across channels as scores.
    # You can replace it with your own scoring function if needed.
    return matrix.mean(axis=0)

def dp_sampling(matrix, epsilon, sample_size):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    # Flatten the scores and probabilities for sampling
    flat_scores = scores.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)

    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_scores)),
        size=sample_size,
        replace=False,
        p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)

    # Set the sampled elements to their original values for all channels
    for channel in range(matrix.shape[0]):
        np.put(output_matrix[channel], sampled_indices, matrix[channel].flatten()[sampled_indices])

    return output_matrix

matrix = np.random.rand(3, 3, 3)
epsilon = 1.0
sample_size = 8

sampled_matrix = dp_sampling(matrix, epsilon, sample_size)
print(sampled_matrix)



for step, (x, y) in enumerate(dataloader_full):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    if args.dataset == 'MNIST':
        x = x.view(x.size(0), -1)
    # print(x)
    # break
    logits_z, logits_y, x_hat, mu, logvar = model(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
    H_p_q = loss_fn(logits_y, y)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    # x = torch.sigmoid(torch.relu(x))
    BCE = reconstruction_function(x_hat, x)  # mse loss

    if learn_model_type == 'vib':
        loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.batch_size * 28 * 28)
    elif learn_model_type == 'nips':
        loss = H_p_q

    optimizer.zero_grad()
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += torch.sum(param.grad ** 2)
    grad_norm = torch.sqrt(grad_norm).item()
    print(f"Gradient L2-norm: {grad_norm}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
    optimizer.step()