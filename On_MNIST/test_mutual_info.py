import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MINE(nn.Module):
    def __init__(self, output_size, input_size, hidden_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size + output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        xy = torch.cat([x, y], dim=1)
        # print(xy.shape)
        h = torch.relu(self.fc1(xy))
        return self.fc2(h)


def train_mine(mine, dataloader, optimizer, alpha=0.1):
    mine.train()
    for x, y in dataloader:
        batch_size = x.shape[0]
        x_shuffle = x[torch.randperm(batch_size)]
        y_shuffle = y[torch.randperm(batch_size)]
        xy = mine(x, y)
        x_y_shuffle = mine(x, y_shuffle)
        x_shuffle_y = mine(x_shuffle, y)
        loss = -(torch.mean(xy) - torch.log(torch.mean(torch.exp(x_y_shuffle))) - torch.log(
            torch.mean(torch.exp(x_shuffle_y))) + alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


# 示例数据
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
Y = torch.tensor([[9], [6], [4]], dtype=torch.float)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# 训练 MINE 算法
mine = MINE(output_size=1, input_size=3, hidden_size=64)
optimizer = optim.Adam(mine.parameters(), lr=1e-3)
for epoch in range(100):
    loss = train_mine(mine, dataloader, optimizer)
    print(f"Epoch {epoch + 1}, loss={loss:.4f}")

# 计算互信息
x = X.view(X.shape[0], -1)
y = Y.view(Y.shape[0], -1)

print(x)
with torch.no_grad():
    xy = mine(x, y)
    x_y_shuffle = mine(x, y[torch.randperm(3)])
    x_shuffle_y = mine(x[torch.randperm(3)], y)
    mi = torch.mean(xy) - torch.log(torch.mean(torch.exp(x_y_shuffle))) - torch.log(torch.mean(torch.exp(x_shuffle_y)))
    print(f"Mutual information: {mi.item():.4f}")
