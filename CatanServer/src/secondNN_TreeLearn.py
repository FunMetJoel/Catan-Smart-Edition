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

import torch.optim as optim
import firstTree
import secondNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.ion()

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
    if len(winnersPoints_t) >= 25:
        means = winnersPoints_t.unfold(0, 25, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(12), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
if __name__ == "__main__":
    if input("Train model? (y/n): ") != "y":
        exit()
        
    startDateTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"model-2fromTree-{startDateTime}.pth"
        
    ai = secondNN.secondNN()
    ai.model.to(device)
    
    optimizer = optim.Adam(ai.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    correctAwnserBot = firstTree.firstTreeBot()
    
    gamesPlayed = 0
    maxRounds = 1000
    while True:
        # play a game and learn from it
        catanGame = catanData.CatanState()
        catanGame.setupRandomly()
        
        for i in range(maxRounds):
            bestMove = correctAwnserBot.getBestAction(catanGame)
            bestAction = bestMove[0]
            
            hexes, players = ai.interpret_state(catanGame)
        
            hexes = hexes.to(device)
            players = players.to(device)
            
            with torch.no_grad():
                actions, roads, settlements, tradefor, tradewidth = ai.model(hexes, players)
                
            
        gamesPlayed += 1
    