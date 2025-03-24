from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime as dt

import catanData
import abstractCatanBot
from compiledCordinateSystem import compiledHexIndex, compiledEdgeIndex, compiledCornerIndex
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import sys

# line profiler
# import line_profiler

# Create a tensor and move it to GPU
x = torch.randn(10000, 10000).cuda()
print(x.device)  # Should show 'cuda:0'


# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example dataset (replace with your data)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Create Dataset and DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class fithNN(abstractCatanBot.CatanBot):
    def __init__(self, model:CatanNN):
        self.model = model
        self.model.to(device)
    
    def __init__(self):
        self.model:CatanNN = CatanNN()
        self.model.to(device)
        
    def normalizePlayer(self, player, currentPlayer):
        if player == 0:
            return 0
        elif player == currentPlayer:
            return 1
        else:
            return -1
        
    # @line_profiler.profile
    def interpret_state(self, game_state:catanData.CatanState):
        hextensors = np.zeros((19, 25))
        playertensor = np.zeros(6)
        
        
        for i in range(19):
            hex = game_state.hexes[i]
            hextensors[i][0] = (hex[0] == 0) # Resource wood
            hextensors[i][1] = (hex[0] == 1) # Resource brick
            hextensors[i][2] = (hex[0] == 2) # Resource wheat
            hextensors[i][3] = (hex[0] == 3) # Resource sheep
            hextensors[i][4] = (hex[0] == 4) # Resource ore
            hextensors[i][5] = (6-abs(hex[1]-7))/36 # Dice roll chance
            hextensors[i][6] = hex[2] # Has robber
            cornerInxexes = compiledHexIndex.neighbourCorners[i]
            corners = []
            for cornerIndex in cornerInxexes:
                corner = game_state.corners[cornerIndex]
                corners.append(corner)
            edgeIndexes = compiledHexIndex.neighbourEdges[i]
            edges = []
            for edgeIndex in edgeIndexes:
                edge = game_state.edges[edgeIndex]
                edges.append(edge)
            hextensors[i][7] = self.normalizePlayer(corners[0][0], game_state.currentPlayer)
            hextensors[i][8] = corners[0][1]
            hextensors[i][9] = self.normalizePlayer(corners[1][0], game_state.currentPlayer)
            hextensors[i][10] = corners[1][1]
            hextensors[i][11] = self.normalizePlayer(corners[2][0], game_state.currentPlayer)
            hextensors[i][12] = corners[2][1]
            hextensors[i][13] = self.normalizePlayer(corners[3][0], game_state.currentPlayer)
            hextensors[i][14] = corners[3][1]
            hextensors[i][15] = self.normalizePlayer(corners[4][0], game_state.currentPlayer)
            hextensors[i][16] = corners[4][1]
            hextensors[i][17] = self.normalizePlayer(corners[5][0], game_state.currentPlayer)
            hextensors[i][18] = corners[5][1]
            hextensors[i][19] = self.normalizePlayer(edges[0][0], game_state.currentPlayer)
            hextensors[i][20] = self.normalizePlayer(edges[1][0], game_state.currentPlayer)
            hextensors[i][21] = self.normalizePlayer(edges[2][0], game_state.currentPlayer)
            hextensors[i][22] = self.normalizePlayer(edges[3][0], game_state.currentPlayer)
            hextensors[i][23] = self.normalizePlayer(edges[4][0], game_state.currentPlayer)
            hextensors[i][24] = self.normalizePlayer(edges[5][0], game_state.currentPlayer)            
            
        
        for j in range(5):
            playertensor[j] = float(game_state.players[game_state.currentPlayer][j])
            
        playertensor[5] = float(game_state.points[game_state.currentPlayer])
                    
        hextensors = torch.tensor(hextensors, dtype=torch.float)
        playertensor = torch.tensor(playertensor, dtype=torch.float)
        return hextensors, playertensor
    
    def make_move(self, game_state):
        hexes, player = self.interpret_state(game_state)
        
        hexes = hexes.to(device)
        player = player.to(device)
        
        with torch.no_grad():
            actions, roads, settlements, tradefor, tradewidth = self.model(hexes, player)
            
        actions = actions.numpy()
        roads = roads.numpy()
        settlements = settlements.numpy()
        tradefor = tradefor.numpy()
        tradewidth = tradewidth.numpy()
        
        actionMask = game_state.getActionMask()
        
        for i in range(len(actions)):
            if actionMask[i] == 0:
                actions[i] = 0
        
        actions = np.argmax(actions)
        if actions == 0:
            game_state.endTurn()
            # print("End turn")
        elif actions == 1:
            roadMask = game_state.getEdgeMask()
            roads[~roadMask] = -np.inf
            road = np.argmax(roads)
            game_state.buildRoad(road)
            # print("Build road")
        elif actions == 2:
            settlementMask = game_state.getCornerMask()
            settlements[~settlementMask] = -np.inf
            settlement = np.argmax(settlements)
            game_state.buildSettlement(settlement)
            print("\t Build settlement (+1) at", settlement)
        elif actions == 3:
            cityMask = game_state.getCityMask()
            settlements[~cityMask] = -np.inf
            city = np.argmax(settlements)
            game_state.buildCity(city)
            print("\t Build city (+2) at", city)
        elif actions == 4:
            tradeFor = np.argmax(tradefor)
            tradewidthMask = game_state.getTradeWithBankMask()
            tradewidth[~tradewidthMask] = -np.inf
            tradeWith = np.argmax(tradewidth)
            game_state.tradeWithBank(tradeWith, tradeFor)
            
            # print("Trade", tradeFor, "for 4x", tradeWith)
            
    def make_opening_move(self, game_state):
        # place settlement
        hexes, player = self.interpret_state(game_state)
        
        hexes = hexes.to(device)
        player = player.to(device)
        
        with torch.no_grad():
            actions, roads, settlements, tradefor, tradewidth = self.model(hexes, player)
            
        settlementMask = game_state.getCornerMask()
        settlements[~settlementMask] = -np.inf
        settlement = np.argmax(settlements)
        game_state.buildSettlement(settlement)
        
        # place road
        hexes, player = self.interpret_state(game_state)
        
        hexes = hexes.to(device)
        player = player.to(device)
        
        with torch.no_grad():
            actions, roads, settlements, tradefor, tradewidth = self.model(hexes, player)
            
        roadMask = game_state.getEdgeMask()
        roads[~roadMask] = -np.inf
        road = np.argmax(roads)
        game_state.buildRoad(road)


class CatanNN(nn.Module):
    def __init__(self):
        super(CatanNN, self).__init__()
        # 19 hexes, 25 values per hex
        self.h1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(25, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])
        self.h2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])
        self.h3 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), 
                nn.ReLU()
            ) for _ in range(19)
        ])
        self.h4 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])
        
        # 6 values per player
        self.p1 = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 19 + 64, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256*2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256*2, 256*4),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256*4, 256*8),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256*8, 256*4),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256*4, 256*2),
            nn.ReLU()
        )
        
        self.actionHead = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.Sigmoid()
        )
        self.outputAction = nn.Sequential(
            # 10000 = end turn, 01000 = build settlement, 00100 = build city, 00010 = build road, 00001 = trade
            nn.Linear(256, 5),
            nn.Sigmoid()
        )
        
        self.roadsHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.outputRoads = nn.Sequential(
            nn.Linear(256, 72),
            nn.Sigmoid()
        )
        
        self.settlementsHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.outputSettlements = nn.Sequential(
            nn.Linear(256, 54),
            nn.Sigmoid()
        )
        
        self.citiesHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.outputCities = nn.Sequential(
            nn.Linear(256, 54),
            nn.Sigmoid()
        )
        
        self.tradeForHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.tradeFor = nn.Sequential(
            nn.Linear(256, 5),
            nn.Sigmoid()
        )
        
        self.tradeAwayHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.tradeAway = nn.Sequential(
            nn.Linear(256, 5),
            nn.Sigmoid()
        )
        
        self.robberHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.robber = nn.Sequential(
            nn.Linear(256, 19),
            nn.Sigmoid()
        )
        
    def forward(self, input_hexes, input_player):
        # print("input_hexes", input_hexes.shape)
        # Process each layer in a batch-wise manner
        processed_hexes = [head(input_hexes[:, i, :]) for i, head in enumerate(self.h1)]
        processed_hexes = [head(processed_hexes[i]) for i, head in enumerate(self.h2)]
        processed_hexes = [head(processed_hexes[i]) for i, head in enumerate(self.h3)]
        processed_hexes = [head(processed_hexes[i]) for i, head in enumerate(self.h4)]
        processed_hexes = torch.cat(processed_hexes, dim=1)  # Concatenating along the feature dimension
        # print("processed_hexes", processed_hexes.shape)
        
        # Process player input in a batch-wise manner
        # print("input_player", input_player.shape)
        processed_player = self.p1(input_player)
        # print("processed_player", processed_player.shape)
        processed_player = self.p2(processed_player)
        processed_player = self.p3(processed_player)
        processed_player = self.p4(processed_player)
        # print("processed_player", processed_player.shape)
        
        # Concatenating along the feature dimension (dim=1 instead of dim=0)
        combined = torch.cat([processed_hexes, processed_player], dim=1) 
        # print("combined", combined.shape)
        
        fc = self.fc1(combined)
        fc = self.fc2(fc)
        fc = self.fc3(fc)
        fc = self.fc4(fc)
        fc = self.fc5(fc)
        fc = self.fc6(fc)
        # print("fc", fc.shape)
        
        action = self.actionHead(fc)
        action = self.outputAction(action)
        
        roads = self.roadsHead(fc)
        roads = self.outputRoads(roads)
        
        settlements = self.settlementsHead(fc)
        settlements = self.outputSettlements(settlements)
        
        cities = self.citiesHead(fc)
        cities = self.outputCities(cities)
        
        tradeFor = self.tradeForHead(fc)
        tradeFor = self.tradeFor(tradeFor)
        
        tradeAway = self.tradeAwayHead(fc)
        tradeAway = self.tradeAway(tradeAway)
        
        robber = self.robberHead(fc)
        robber = self.robber(robber)
        
        # print("action", action.shape)
        # print("roads", roads.shape)
        # print("settlements", settlements.shape)
        # print("cities", cities.shape)
        # print("tradeFor", tradeFor.shape)
        # print("tradeAway", tradeAway.shape)
        # print("robber", robber.shape)
        
        return action, roads, settlements, cities, tradeFor, tradeAway, robber
    
class Plotter:
    def __init__(self):
        self.values = {}
        plt.figure(1)
        
    def initExample(self):
        self.values = {
            "score": [],
        }
        
    def add(self, key, value):
        if key not in self.values:
            self.values[key] = []
        self.values[key].append(value)
        
    def plot(self):
        plt.clf()
        for key, value in self.values.items():
            plt.plot(value, label=key)
            
# Define loss functions
criterion_action = nn.BCELoss()  # Multi-label classification
criterion_roads = nn.BCELoss()
criterion_settlements = nn.BCELoss()
criterion_cities = nn.BCELoss()
criterion_tradeFor = nn.BCELoss()
criterion_tradeAway = nn.BCELoss()
criterion_robber = nn.BCELoss()

def compute_loss(outputs, targets):
    action, roads, settlements, cities, tradeFor, tradeAway, robber = outputs
    target_action, target_roads, target_settlements, target_cities, target_tradeFor, target_tradeAway, target_robber = targets
    
    # target_action = target_action.to(device).float()
    # target_roads = target_roads.to(device).float()
    target_settlements = target_settlements.to(device).float()
    # target_cities = target_cities.to(device).float()
    # target_tradeFor = target_tradeFor.to(device)
    # target_tradeAway = target_tradeAway.to(device)
    # target_robber = target_robber.to(device).float()
    
    # loss_action = criterion_action(action, target_action)
    # loss_roads = criterion_roads(roads, target_roads)
    loss_settlements = criterion_settlements(settlements, target_settlements)
    # loss_cities = criterion_cities(cities, target_cities)
    # loss_tradeFor = criterion_tradeFor(tradeFor, target_tradeFor)
    # loss_tradeAway = criterion_tradeAway(tradeAway, target_tradeAway)
    # loss_robber = criterion_robber(robber, target_robber)

    # Weighted sum of losses (you can adjust the weights)
    # total_loss = (loss_action + loss_roads + loss_settlements +
    #               loss_cities + loss_tradeFor + loss_tradeAway + loss_robber) / 7
    
    total_loss = loss_settlements

    return total_loss


shouldTrain = input("Train catanmodel? (y/n): ")
if shouldTrain == "y":

    class CatanDataset(Dataset):
        def __init__(self, file_path):
            self.data = torch.load(file_path)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            return sample['input_hexes'], sample['input_player'], sample['targets']

    # Load dataset
    dataset = CatanDataset("D:\AiTrainingData-Catan\dataset-openings-2025-03-12-17-35-21.pth.backup2025-03-13-17-55-04.pth")    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Initialize model and move to GPU
    model = CatanNN().to(device)
    
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1)
    
    num_epochs = 10000
    plotter = Plotter()
    
    # @line_profiler.profile
    def train_model():
        for epoch in range(num_epochs):
            for hexes, player, targets in dataloader:
                # Move inputs and targets to GPU
                hexes, player = hexes.to(device), player.to(device)
                
                for target in targets:
                    target = target.to(device)
                
                # Forward pass
                outputs = model(hexes, player)
                loss = compute_loss(outputs, targets)
                
                # Take the log of the loss
                plotter.add("log(score)", torch.log(loss).item())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            plotter.plot()
            plt.pause(0.001) 
            
            sys.stdout.write(f"\r Epoch {epoch}")
            sys.stdout.flush()
                
            if (epoch + 1) % 10 == 0:
                # sys.stdout.write(f"\r Records {DatasetNum}")
                # sys.stdout.flush()
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                try:
                    torch.save(model, f"D:\AiTrainingData-Catan\catanModel-v4-{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{epoch}.pth")
                except Exception as e:
                    print("Failed to save model")
                    print(e)
                    pass

    train_model()
    # save model
    torch.save(model, f"D:\AiTrainingData-Catan\catanModel-v4-{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth")        
        
    while True:
        plotter.plot()
        plt.pause(0.001)

shouldTrain = input("Train rekenModel? (y/n): ")
if shouldTrain == "y":
    # Define the Neural Network
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.hidden = nn.Linear(1, 10)  # Input to hidden layer
            self.output = nn.Linear(10, 1)  # Hidden to output layer

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = self.output(x)
            return x

    # Initialize model and move to GPU
    model = SimpleNN().to(device)
    catanModel = CatanNN().to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Training loop
    num_epochs = 100000
    plotter = Plotter()
    plotter.initExample()

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Move inputs and targets to GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Take the log of the loss
            plotter.add("log(score)", torch.log(loss).item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            plotter.plot()
            plt.pause(0.001)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[5.0]], dtype=torch.float32).to(device)
        prediction = model(test_input)
        print(f'Prediction for input 5.0: {prediction.item()}')
