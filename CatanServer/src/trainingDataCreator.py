from torch import tensor
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import datetime as dt

import firstTree as ft
import secondTree as st
import fithNN as fnn

import catanData as Catan

import argparse

FT = ft.firstTreeBot()

tree = st.secondTreeBot()
nn = fnn.fithNN()


import os
import sys



filePath = f"D:\AiTrainingData-Catan\dataset{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth"
openingFilePath = f"D:\AiTrainingData-Catan\dataset-openings-{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth"

openingsOnly = False
parser = argparse.ArgumentParser()
parser.add_argument("--openingsOnly", help="Only collect openings", action="store_true")
args = parser.parse_args()
if args.openingsOnly:
    openingsOnly = True


catanGame = Catan.CatanState()

dataset = []
openingMoveDataset = []

def mainloop():
    global catanGame
        
    setupBoard()
    
    lastPlayer = -1
    while True:
        if openingsOnly:
            setupBoard()
            continue
        
        print(catanGame.currentPlayer)
        if lastPlayer != catanGame.currentPlayer:
            print(catanGame.currentPlayer, )
            lastPlayer = catanGame.currentPlayer
        else:  
            generateData(catanGame, dataset, filePath)
            
        if catanGame.round % 10 == 0:
            generateData(catanGame, dataset, filePath)
            
        FT.make_move(catanGame)
        
        if catanGame.points.max() >= 10:
            # save game data
            setupBoard()
             
def setupBoard():
    global catanGame
    catanGame = Catan.CatanState()
    
    while catanGame.round < 2:
        
        generateData(catanGame, openingMoveDataset, openingFilePath)
        
        tree.make_opening_move(catanGame)      
        catanGame.endTurn()

def generateData(catanGame, dataset, filePath):
    # get inputs
    hexTensors, playerTensor = nn.interpret_state(catanGame)
    
    # action, roads, settlements, cities, tradeFor, tradeAway, robber
    # get outputs

    maxBuildRoadScore = -1000
    roadScores = np.zeros(72)
    for i in range(72):
        newGameState = catanGame.copy()
        newGameState.buildRoad(i)
        score = tree.interpretState(newGameState)
        roadScores[i] = score
        if score > maxBuildRoadScore:
            maxBuildRoadScore = score
            
    maxBuildSettlementScore = -1000
    settlementScores = np.zeros(54)
    for i in range(54):
        newGameState = catanGame.copy()
        newGameState.buildSettlement(i)
        score = tree.interpretState(newGameState)
        settlementScores[i] = score
        if score > maxBuildSettlementScore:
            maxBuildSettlementScore = score
            
    maxBuildCityScore = -1000
    cityScores = np.zeros(54)
    for i in range(54):
        newGameState = catanGame.copy()
        newGameState.buildCity(i)
        score = tree.interpretState(newGameState)
        cityScores[i] = score
        if score > maxBuildCityScore:
            maxBuildCityScore = score
            
    tradeForScores = np.zeros(19)
    tradeAwayScores = np.zeros(19)
    
    robberScores = np.zeros(19)
    for i in range(19):
        newGameState = catanGame.copy()
        newGameState.moveRobber(i)
        score = tree.interpretState(newGameState)
        robberScores[i] = score
        
        
    maxScore = max(maxBuildRoadScore, maxBuildSettlementScore, maxBuildCityScore, 0.001)
    
    # normalize all scores
    roadScores = roadScores / maxScore
    settlementScores = settlementScores / maxScore
    cityScores = cityScores / maxScore
    robberScores = robberScores / maxScore
    
    
    actionScores = np.zeros(5)
    actionScores[0] = 0
    actionScores[1] = maxBuildRoadScore / maxScore
    actionScores[2] = maxBuildSettlementScore / maxScore
    actionScores[3] = maxBuildCityScore / maxScore
    actionScores[4] = 0
    
    
    targets = (
        actionScores,     # action target (binary)
        roadScores,    # roads target (binary)
        settlementScores,    # settlements target (binary)
        cityScores,    # cities target (binary)
        torch.rand(5),    # tradeFor target (binary)
        torch.rand(5),    # tradeAway target (binary)
        robberScores     # robber target (binary)
    )
    
    dataset.append({
        'input_hexes': hexTensors, 
        'input_player': playerTensor, 
        'targets': targets
    })
        
    
    DatasetNum = len(dataset)
    sys.stdout.write(f"\r Records {DatasetNum}")
    sys.stdout.flush()
    
    #make backup every 1000
    if (DatasetNum % 10000) == 0:
        backupfilepath = filePath + f".backup{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth"
        torch.save(dataset, backupfilepath)
        torch.save(dataset, filePath)

        
        

    
if __name__ == "__main__":
    mainloop()