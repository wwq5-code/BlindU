import torch
import torch.nn as nn
import torch.optim as optim

class MINE(nn.Module):
    def __init__(self, input_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mutual_information_estimator(x1, x2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x1 = torch.tensor(x1, dtype=torch.float32).to(device)
    x2 = torch.tensor(x2, dtype=torch.float32).to(device)

    input_size = x1.shape[1] + x2.shape[1]
    mine = MINE(input_size).to(device)

    optimizer = optim.Adam(mine.parameters(), lr=0.01)

    for step in range(100):
        # Joint samples
        x_joint = torch.cat((x1, x2), dim=1)

        # Marginal samples
        x1_marginal = x1[torch.randperm(len(x1))]
        x2_marginal = x2[torch.randperm(len(x2))]
        x_marginal = torch.cat((x1_marginal, x2_marginal), dim=1)

        # Training step
        optimizer.zero_grad()
        joint_scores = mine(x_joint)
        marginal_scores = mine(x_marginal)
        mi_loss = -(torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores))))
        mi_loss.backward()
        optimizer.step()
        print(mi_loss.item())

    mi_estimate = -(torch.mean(mine(x_joint)) - torch.log(torch.mean(torch.exp(mine(x_marginal)))).item())
    return mi_estimate


x1 = [[1], [2], [3], [4], [5]]
x2 = [[2], [3], [4], [5], [6]]

mi = mutual_information_estimator(x1, x2)
print(f'Mutual Information Estimate: {mi}')
