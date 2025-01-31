from __future__ import annotations

import fourthNN
import firstTree
import numpy as np
import random
import catanData as Catan
import torch
import datetime
import matplotlib.pyplot as plt

bestScores = []
def plotbestScore():
    plt.figure(1)
    plt.clf()
    plt.title(f'Training... (from model-2fromTree)')
    plt.xlabel('Episode')
    plt.ylabel('bestScores')
    plt.plot(bestScores)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == "__main__":
    ais = [fourthNN.forthNN() for i in range(4)]
    for i in range(4):
        ais[i].model.load_state_dict(torch.load(f"src/bots/model-2fromTree/2025-01-23-21-51-41/model4NN-106-14400.pth"))
    
    filename = f"model-2fromTree"
    folder = f"src/bots/model-2fromTree/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # create folder
    import os
    os.makedirs(folder)
    
    scores = [0, 0, 0, 0]
    
    tree = firstTree.firstTreeBot()
        
    if input("train? (y/n): ") != "y":
        exit()
        
    episodes = 0
        
    while True:
        
        for i in range(4):
            catanGame = Catan.CatanState()
            while catanGame.round < 2:
                tree.make_opening_move(catanGame)
                catanGame.endTurn()
                
        
            maxrounds = 100        
            while catanGame.points.max() < 10:
                if catanGame.round >= maxrounds:
                    break
                if catanGame.currentPlayer == i:
                    ai = ais[i]
                    ai.make_move(catanGame)
                else:
                    tree.make_move(catanGame)
                    
            scores[i] += catanGame.points[i]
            
        episodes += 1
        
        if episodes % 25 == 0:
            best = np.argmax(scores)
            secondBest = np.argmax([scores[i] for i in range(4) if i != best])
            print(f"Episode {episodes}: {scores} best: {best}")
            for i in range(4):
                if i != best and i != secondBest:
                    ais[i].model.load_state_dict(ais[best].model.state_dict())
                    #mutate
                    for param in ais[i].model.parameters():
                        param.data += 0.1 * torch.randn_like(param)
                        
                        
            if episodes % 100 == 0:
                # save best model
                try:
                    torch.save(ais[best].model.state_dict(), f"{folder}/model4NN-{scores[best]}-{episodes}.pth")
                except:
                    print("error saving model")
            
            bestScores.append(scores[best])
            plotbestScore()
            scores = [0, 0, 0, 0]

                        
                        
            
            
        
        