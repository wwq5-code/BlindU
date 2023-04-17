import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the STL-10 dataset
transform = transforms.Compose(
    [transforms.Resize((100, 100)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5000, shuffle=False)

#find birds

for data in trainloader:
    images, labels = data
    print(images.shape)
    print(labels)
    for i in range(725,730):
        if labels[i]!=1: continue
        print(i)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()
    break

# Select an image
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
img = images[729].permute(1, 2, 0).numpy()

# Divide the image into a 5x5 grid
grid_size = 5
cell_size = img.shape[0] // grid_size

# Fill some grids with grey
grey_color = 0.5
filled_grids = [(0, 0), (0, 1), (0, 4),
                (1, 0), (1, 1), (1, 2), (1, 4),
                (2, 0), (2, 3), (2, 4),
                (3, 0), (3, 1), (3, 2), (3, 4),
                (4, 1), (4, 2), (4, 4),]
all_grids = [(x, y) for x in range(0, 5) for y in range(0, 5)]

num_grids_to_select = 17

random_selected_grids = random.sample(all_grids, num_grids_to_select)

print(random_selected_grids)

for i, j in filled_grids:
    img[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] = grey_color

# Visualize the modified image
plt.imshow(img)
plt.axis('off')
plt.show()


grid_size = 5
grid_img_size = img.shape[0] // grid_size

# Add thin white lines between grid cells
for i in range(1, grid_size):
    img[i * grid_img_size - 1, :, :] = 1
    img[:, i * grid_img_size - 1, :] = 1

# Display the modified image with grid lines
plt.imshow(img)
plt.axis('off')
plt.show()