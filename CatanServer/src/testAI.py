import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Part 1: Simple AI program that adds two numbers
def add_numbers(a, b):
    return torch.tensor(a) + torch.tensor(b)

# Example usage
num1, num2 = 5, 7
print("Sum:", add_numbers(num1, num2).item())

# Part 2: Trainer program using Dataset and DataLoader
class SimpleDataset(Dataset):
    def __init__(self):
        self.data = [(torch.tensor([x], dtype=torch.float32), torch.tensor([2 * x + 1], dtype=torch.float32)) for x in range(100)]
        print(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Define a simple linear model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Training function
def train_model():
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(20):
        for inputs, targets in dataloader:
            print("Inputs", inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

# Train and get the model
trained_model = train_model()
