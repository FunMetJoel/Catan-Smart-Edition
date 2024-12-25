from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import catanData
import abstractCatanBot
from compiledCordinateSystem import compiledHexIndex, compiledEdgeIndex, compiledCornerIndex
import random

import matplotlib
import matplotlib.pyplot as plt
import datetime
# import line_profiler
# import os

# os.environ["LINE_PROFILE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.ion()

class firstNN(abstractCatanBot.CatanBot):
    def __init__(self, model:CatanAiModel):
        self.model = model
        self.model.to(device)
    
    def __init__(self):
        self.model:CatanAiModel = CatanAiModel()
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
        hextensors = np.zeros((19, 21))
        playertensors = np.zeros((4, 5))
        
        
        for i in range(19):
            hex = game_state.hexes[i]
            hextensors[i][0] = hex[0] # Resource
            hextensors[i][1] = hex[1] # Dice roll
            hextensors[i][2] = hex[2] # Has robber
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
            hextensors[i][3] = self.normalizePlayer(corners[0][0], game_state.currentPlayer)
            hextensors[i][4] = corners[0][1]
            hextensors[i][5] = self.normalizePlayer(corners[1][0], game_state.currentPlayer)
            hextensors[i][6] = corners[1][1]
            hextensors[i][7] = self.normalizePlayer(corners[2][0], game_state.currentPlayer)
            hextensors[i][8] = corners[2][1]
            hextensors[i][9] = self.normalizePlayer(corners[3][0], game_state.currentPlayer)
            hextensors[i][10] = corners[3][1]
            hextensors[i][11] = self.normalizePlayer(corners[4][0], game_state.currentPlayer)
            hextensors[i][12] = corners[4][1]
            hextensors[i][13] = self.normalizePlayer(corners[5][0], game_state.currentPlayer)
            hextensors[i][14] = corners[5][1]
            hextensors[i][15] = self.normalizePlayer(edges[0][0], game_state.currentPlayer)
            hextensors[i][16] = self.normalizePlayer(edges[1][0], game_state.currentPlayer)
            hextensors[i][17] = self.normalizePlayer(edges[2][0], game_state.currentPlayer)
            hextensors[i][18] = self.normalizePlayer(edges[3][0], game_state.currentPlayer)
            hextensors[i][19] = self.normalizePlayer(edges[4][0], game_state.currentPlayer)
            hextensors[i][20] = self.normalizePlayer(edges[5][0], game_state.currentPlayer)            
            
        for i in range(4):
            playerdata = game_state.players[i]
                    
            for j in range(5):
                playertensors[i][j] = float(playerdata[j])
                    
        hextensors = torch.tensor(hextensors, dtype=torch.float)
        playertensors = torch.tensor(playertensors, dtype=torch.float)
        return hextensors, playertensors
    
    def make_move(self, game_state):
        hexes, players = self.interpret_state(game_state)
        
        hexes = hexes.to(device)
        players = players.to(device)
        
        with torch.no_grad():
            actions, roads, settlements, tradefor, tradewidth = self.model(hexes, players)
            
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
            roads = roads * roadMask
            road = np.argmax(roads)
            game_state.buildRoad(road)
            # print("Build road")
        elif actions == 2:
            settlementMask = game_state.getCornerMask()
            settlements = settlements * settlementMask
            settlement = np.argmax(settlements)
            game_state.buildSettlement(settlement)
            print("Build settlement (+1) at", settlement)
        elif actions == 3:
            cityMask = game_state.getCityMask()
            settlements = settlements * cityMask
            city = np.argmax(settlements)
            game_state.buildCity(city)
            print("Build city (+2) at", city)
        elif actions == 4:
            tradeFor = np.argmax(tradefor)
            tradewidthMask = game_state.getTradeWithBankMask()
            tradewidth = tradewidth * tradewidthMask
            tradeWith = np.argmax(tradewidth)
            game_state.tradeWithBank(tradeWith, tradeFor)
            
            # print("Trade", tradeFor, "for 4x", tradeWith)
        
class CatanAiModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 19 hexes, 21 values each and 4 players, 6 values each
        # Define the heads for the 21-size input vectors
        self.hexes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(21, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])

        # Define the heads for the 6-size input vectors
        self.players = nn.ModuleList([
            nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # Example output processing layer, combining all head outputs
        self.fullyConnected1 = nn.Sequential(
            nn.Linear(19 * 64 + 4 * 32, 128*32),
            nn.ReLU()
        )
        
        self.fullyConnected2 = nn.Sequential(
            nn.Linear(128*32, 64*32),
            nn.ReLU()
        )
        
        self.actionHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.roadsHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.settlementsHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.tradeForHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.tradeWithHead = nn.Sequential(
            nn.Linear(64*32, 64*32),
            nn.ReLU()
        )
        
        self.outputAction = nn.Sequential(
            # 10000 = end turn, 01000 = build settlement, 00100 = build city, 00010 = build road, 00001 = trade
            nn.Linear(64*32, 5),
            nn.Sigmoid()
        )
        
        self.outputRoads = nn.Sequential(
            nn.Linear(64*32, 72),
            nn.Sigmoid()
        )
        
        self.outputSettlements = nn.Sequential(
            nn.Linear(64*32, 54),
            nn.Sigmoid()
        )
        
        self.outputTradeFor = nn.Sequential(
            nn.Linear(64*32, 5),
            nn.Sigmoid()
        )
        
        self.OutputTradeWith = nn.Sequential(
            nn.Linear(64*32, 5),
            nn.Sigmoid()
        )
        
    def forward(self, input_hexes, input_players):
        # Apply each head to the corresponding vector in input_21
        processed_hexes = [head(vector) for head, vector in zip(self.hexes, input_hexes)]
        processed_hexes = torch.cat(processed_hexes, dim=-1)  # Concatenate along the feature dimension

        # Apply each head to the corresponding vector in input_6
        processed_players = [head(vector) for head, vector in zip(self.players, input_players)]
        processed_players = torch.cat(processed_players, dim=-1)  # Concatenate along the feature dimension

        # Combine all processed outputs
        combined = torch.cat([processed_hexes, processed_players], dim=-1)

        # Final output processing
        w = self.fullyConnected1(combined)
        w = self.fullyConnected2(w)
        x = self.actionHead(w)
        y = self.roadsHead(w)
        z = self.settlementsHead(w)
        a = self.tradeForHead(w)
        b = self.tradeWithHead(w)
        
        action = self.outputAction(x)
        roads = self.outputRoads(y)
        settlements = self.outputSettlements(z)
        tradeFor = self.outputTradeFor(a)
        tradeWith = self.OutputTradeWith(b)
        
        return action, roads, settlements, tradeFor, tradeWith
    
winnersPoints = []
roundActions = []

def plot_durations(show_result=False):
    plt.figure(1)
    winnersPoints_t = torch.tensor(winnersPoints, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('winnersPoints')
    plt.plot(winnersPoints_t.numpy())
    plt.plot(roundActions)
    # Take 100 episode averages and plot them too
    if len(winnersPoints_t) >= 100:
        means = winnersPoints_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    
if __name__ == "__main__":
    if input("Train model? (y/n): ") != "y":
        exit()
        
    startDateTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    bots = [firstNN(), firstNN(), firstNN(), firstNN()]
    totalScore = [0, 0, 0, 0]
    
    episode = 0
    while True:
        episode += 1
        game = catanData.CatanState()
        
        for i in range(4):
            for j in range(2):
                cornerMask = game.getCornerMask(i)
                randomIndex = np.random.randint(52)
                while not cornerMask[randomIndex]:
                    randomIndex = np.random.randint(52)
                game.corners[randomIndex][0] = i+1
                game.corners[randomIndex][1] = 1
                
                edgeMask = game.getEdgeMask(i)
                ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
                while not edgeMask[ramdomRoadIndex]:
                    ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
                game.edges[ramdomRoadIndex][0] = i+1
                game.edges[ramdomRoadIndex][1] = 1
                
        game.round = 2
        
        for i in range(1000):
            currentBot = bots[game.currentPlayer]
            currentBot.make_move(game)
            # print(game.currentPlayer, game.round)
            # print(game.players[game.currentPlayer])
            if game.hasGameEnded():
                roundActions.append(i/100)
                break
        else:
            roundActions.append(1000/100)
            
        winnersPoints.append(game.points[np.argmax(game.points)])
        
        for i in range(4):
            totalScore[i] += game.points[i].item()
            
        plot_durations()
        
        print("Episode", episode, "Scores:", totalScore, "Winner:", np.argmax(game.points), "Points:", game.points[np.argmax(game.points)], "rounds:", game.round)
        
        if episode % 25 == 0:
            bestBot = np.argmax(totalScore)
            secondBestBot = np.argsort(totalScore)[-2]
            print("Best bot:", bestBot,"Secondbest bot:", secondBestBot, "Scores:", totalScore)
            torch.save(bots[bestBot].model.state_dict(), f"model{startDateTime}.pth")
            
            newBots:list[firstNN] = [firstNN(), firstNN(), firstNN(), firstNN()]
            newBots[0].model.load_state_dict(bots[bestBot].model.state_dict())
            newBots[1].model.load_state_dict(bots[secondBestBot].model.state_dict())
            newBots[2].model.load_state_dict(bots[bestBot].model.state_dict())
            newBots[3].model.load_state_dict(bots[secondBestBot].model.state_dict())
                
            for i in range(2):
                # Mutate the model
                for param in newBots[i].model.parameters():
                    param.data += 0.1 * torch.randn_like(param)
                    
            bots = newBots
            
            totalScore = [0, 0, 0, 0]

        
        
        

        
        
        
        
    