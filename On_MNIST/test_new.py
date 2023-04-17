import torch
import torch.nn as nn
import torch.optim as optim

class MINE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        hx = torch.relu(self.fc1(x))
        hy = torch.relu(self.fc1(y))
        h = hx + hy
        h = torch.relu(h)
        mi = self.fc2(h)
        return mi

def train_mine(mine, data_loader, optimizer):
    mine.train()
    for (x, y) in data_loader:
        optimizer.zero_grad()
        x = x.float()
        y = y.float()
        batch_size = x.shape[0]
        e1 = mine(x, y)
        e2 = torch.mean(torch.exp(mine(x, torch.roll(y, 1, 0))))
        loss = -(e1.mean() - torch.log(e2))
        loss.backward()
        optimizer.step()
    return loss.item()

input_size = 3
hidden_size = 100
mine = MINE(input_size, hidden_size)
optimizer = optim.Adam(mine.parameters(), lr=0.001)

# generate some fake data
x = torch.randn(100, input_size)
y = torch.randn(100, input_size)

dataset = torch.utils.data.TensorDataset(x, y)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# train the MINE
for epoch in range(10000):
    loss = train_mine(mine, data_loader, optimizer)
    print(f"Epoch {epoch + 1}, loss={loss:.4f}")

# calculate the mutual information between x and y
with torch.no_grad():
    x = x.float()
    y = y.float()
    mi = mine(x, y)
    print(mi.mean().item())



