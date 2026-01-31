import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = []
        
        for x in range(100):
            for y in range(100):
                # Normalize inputs and outputs
                x_norm = x / 100.0
                y_norm = y / 100.0
                target = (x + y) / 200.0   # Normalize sum
                
                self.data.append((
                    torch.tensor([x_norm, y_norm], dtype=torch.float32),
                    torch.tensor([target], dtype=torch.float32)  # Ensure correct shape
                ))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        print("X", x)
        a = self.linear(x)
        print("A", a)
        return a
    
def train_model(): 
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(20):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()
    print(trained_model)
    
    # Test the trained model
    test_x, test_y = 10, 20
    test_input = torch.tensor([test_x / 100.0, test_y / 100.0], dtype=torch.float32)
    
    with torch.no_grad():
        test_output = trained_model(test_input) * 200  # Rescale back
    
    print("Test output:", test_output.item())
    
    while True:
        x = int(input("Enter x: "))
        y = int(input("Enter y: "))
        
        test_input = torch.tensor([x / 100.0, y / 100.0], dtype=torch.float32)
        
        with torch.no_grad():
            test_output = trained_model(test_input) * 200  # Rescale back
            
        print("Sum:", test_output.item())
