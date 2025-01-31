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
import line_profiler
import os
import firstTree

# os.environ["LINE_PROFILE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.ion()

class forthNN(abstractCatanBot.CatanBot):
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
        hextensors = np.zeros((19, 25))
        playertensor = np.zeros(5)
        
        
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
        
        
class CatanAiModel(nn.Module):
    @line_profiler.profile
    def __init__(self):
        super().__init__()
        # 19 hexes, 21 values each and 4 players, 6 values each
        # Define the heads for the 21-size input vectors
        self.hexes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(25, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])
        
        self.hexes2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(19)
        ])

        # Define the heads for the 6-size input vectors
        self.player = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        self.player2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Example output processing layer, combining all head outputs
        self.fullyConnected1 = nn.Sequential(
            nn.Linear(19 * 64 + 32, 128*32),
            nn.ReLU()
        )
        
        self.fullyConnected2 = nn.Sequential(
            nn.Linear(128*32, 128*32),
            nn.ReLU()
        )
        
        self.fullyConnected3 = nn.Sequential(
            nn.Linear(128*32, 128*32),
            nn.ReLU()
        )
        
        self.fullyConnected4 = nn.Sequential(
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
    
    @line_profiler.profile
    def forward(self, input_hexes, input_players):
        # Apply each head to the corresponding vector in input_21
        processed_hexes = [head(vector) for head, vector in zip(self.hexes, input_hexes)]
        processed_hexes = torch.cat(processed_hexes, dim=-1)  # Concatenate along the feature dimension

        # processed_hexes = [head(vector) for head, vector in zip(self.hexes2, processed_hexes)]
        # processed_hexes = torch.cat(processed_hexes, dim=-1)  # Concatenate along the feature dimension


        # Apply each head to the corresponding vector in input_6
        # processed_players = [head(vector) for head, vector in zip(self.players, input_players)]
        # processed_players = torch.cat(processed_players, dim=-1)  # Concatenate along the feature dimension
        processedPlayer = self.player(input_players)
        processedPlayer = self.player2(processedPlayer)

        # processed_players = [head(vector) for head, vector in zip(self.players2, processed_players)]
        # processed_players = torch.cat(processed_players, dim=-1)  # Concatenate along the feature dimension

        # Combine all processed outputs
        combined = torch.cat([processed_hexes, processedPlayer], dim=-1)

        # Final output processing
        w = self.fullyConnected1(combined)
        w = self.fullyConnected2(w)
        w = self.fullyConnected3(w)
        w = self.fullyConnected4(w)
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
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(f'AI - Training...') # plt.title(f'{startDateTime} - Training...')
    plt.xlabel('Episode')
    plt.ylabel('winnersPoints')
    plt.plot(winnersPoints)
    plt.plot(roundActions)
    pointMeans = [0] * 10
    actionMeans = [0] * 10
    if len(winnersPoints) >= 21:
        for i in range(10, len(winnersPoints)-10):
            pointMeans.append(np.mean(winnersPoints[i-10:i+10]))
            actionMeans.append(np.mean(roundActions[i-10:i+10]))
        plt.plot(pointMeans)
        plt.plot(actionMeans)

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    return pointMeans, actionMeans
    
    
if __name__ == "__main__":
    if input("Train model? (y/n): ") != "y":
        exit()
        
    startDateTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
    bot = forthNN()
    bot.model.load_state_dict(torch.load("src/bots/2025-01-10-22-41-47/model4NN-1.549-2025-01-10-22-41-47.pth"))
    
    bots = [forthNN(), forthNN(), forthNN(), forthNN()]
    
    for i in range(4):
        bots[i].model.load_state_dict(bot.model.state_dict())
    
    totalScore = [0, 0, 0, 0]
    
    tree = firstTree.firstTreeBot()
    
    maxScoreMean = 0
    minActionMean = 10000
    
    episode = 0
    while True:
        episode += 1
        game = catanData.CatanState()
        while game.round < 2:
            tree.make_opening_move(game)
            game.endTurn()
            
        for i in range(1000):
            currentBot = bots[game.currentPlayer]
            currentBot.make_move(game)
            # print(game.currentPlayer, game.round)
            # print(game.players[game.currentPlayer])
            
            # if i % 100 == 0:
            #     print("Actions:", i)
            
            if game.hasGameEnded():
                roundActions.append(i/100)
                break
        else:
            roundActions.append(1000/100)
            
        winnersPoints.append(game.points[np.argmax(game.points)])
        if game.points[np.argmax(game.points)] == 10:
            game.points[np.argmax(game.points)] = 20
        
        for i in range(4):
            totalScore[i] += game.points[i].item()
            
        pointMeans, actionMeans =  plot_durations()
        
        if (actionMeans[-1] < minActionMean) and (actionMeans[-1] > 0):
            minActionMean = actionMeans[-1]
            if not os.path.exists(f"src/bots/{startDateTime}"):
                os.makedirs(f"src/bots/{startDateTime}")
            torch.save(bots[np.argmax(totalScore)].model.state_dict(), f"src/bots/{startDateTime}/model4NN-{minActionMean}-{startDateTime}.pth")
        
        print("Episode", episode, "Scores:", totalScore, "Winner:", np.argmax(game.points), "Points:", game.points[np.argmax(game.points)], "rounds:", game.round)
        
        if episode % 20 == 0:
            bestBot = np.argmax(totalScore)
            secondBestBot = np.argsort(totalScore)[-2]
            print("Best bot:", bestBot,"Secondbest bot:", secondBestBot, "Scores:", totalScore)
            # make directory if it doesn't exist
            
            torch.save(bots[bestBot].model.state_dict(), f"model4NN-{startDateTime}.pth")
            
            newBots:list[forthNN] = [forthNN(), forthNN(), forthNN(), forthNN()]
            newBots[0].model.load_state_dict(bots[bestBot].model.state_dict())
            newBots[1].model.load_state_dict(bots[secondBestBot].model.state_dict())
            newBots[2].model.load_state_dict(bots[bestBot].model.state_dict())
            newBots[3].model.load_state_dict(bots[secondBestBot].model.state_dict())
                
            for i in range(2):
                # Mutate the model
                for param in newBots[i].model.parameters():
                    param.data += 0.1 * torch.randn_like(param)
                    
            for i in range(4):
                newBots[i].model.to(device)
                
            bots = newBots
            
            totalScore = [0, 0, 0, 0]

        
        
        

        
        
        
        
    