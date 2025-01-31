import numpy as np
import compiledCordinateSystem as ccs
import random
from compiledCordinateSystem import compiledCornerIndex, compiledEdgeIndex, compiledHexIndex

import abstractCatanBot as acb
from catanData import CatanState
# import logging
# logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt

class firstTreeBot(acb.CatanBot):
    def __init__(self):
        self.name = "firstTreeBot"
        self.tree = []
        
    def interpretState(self, game_state: CatanState):
        stateScore = 0
        
        # +20 for each point
        stateScore += 20 * game_state.points[game_state.currentPlayer].item()
        
        # if winning move, do it
        if game_state.points[game_state.currentPlayer] >= 10:
            stateScore += 100000
        
        materialCovarage = [0, 0, 0, 0, 0]
        for i in range(54):
            if game_state.corners[i][0] == game_state.currentPlayer + 1:
                for j in compiledCornerIndex.neighbourHexes[i]:
                    if j == -1:
                        continue
                    chanceToRoll = (6-abs(game_state.hexes[j][1].item()-7))/36
                    if game_state.hexes[j][0] == 5:
                        continue
                    materialCovarage[game_state.hexes[j][0]] += chanceToRoll * game_state.corners[i][1].item() # * game_state.hexes[j][2] # TODO: Kijken of deze hier moet
        
        
        # +1 bonus for each material and +amountperroll
        for i in range(5):
            if materialCovarage[i] > 0:
                stateScore += 10
            stateScore += materialCovarage[i] * 100
                
        materials = game_state.players[game_state.currentPlayer]

        # +5 for being able to build a settlement
        if materials[0] >= 1 and materials[1] >= 1 and materials[2] >= 1 and materials[3] >= 1:
            stateScore += 5
            
        # +5 for being able to build a city
        if materials[2] >= 3 and materials[3] >= 2:
            stateScore += 5
        
        # +1 for being able to build a road
        if materials[0] >= 1 and materials[1] >= 1:
            stateScore += 1
            
        # +5 for having a place to build a settlement if round > 2
        if game_state.round > 2:
            cornerMask = game_state.getCornerMask(game_state.currentPlayer)
            for i in range(54):
                if cornerMask[i]:
                    stateScore += 5    
        
        return stateScore
    
    def getPossibleActions(self, game_state: CatanState):
        actionMask = game_state.getActionMask()
        # actionMask[0] is end turn
        # actionMask[1] is build road
        # actionMask[2] is build settlement
        # actionMask[3] is build city
        # actionMask[4] is trade
        
        possbleActions = []
        # possbleActions is a list of tuples
        # (action, score)
        # action is a tuple of the action and the arguments (actionIndex, arguments)
        # score is the score of the action
        
        if actionMask[0]:
            possbleActions.append(((0, None), 0))
            
        if actionMask[1]:
            edgeMask:np.array = game_state.getEdgeMask(game_state.currentPlayer)
            for i in range(72):
                if edgeMask[i]:
                    # calculate score
                    newGameState = game_state.copy()
                    newGameState.buildRoad(i)
                    score = self.interpretState(newGameState)
                    possbleActions.append(((1, i), score))
                    
        if actionMask[2]:
            cornerMask:np.array = game_state.getCornerMask(game_state.currentPlayer)
            for i in range(54):
                if not cornerMask[i]:
                    continue
                # calculate score
                newGameState = game_state.copy()
                newGameState.buildSettlement(i)
                score = self.interpretState(newGameState)
                possbleActions.append(((2, i), score))
                    
        if actionMask[3]:
            cityMask:np.array = game_state.getCityMask(game_state.currentPlayer)
            for i in range(54):
                if not cityMask[i]:
                    continue
                # calculate score
                newGameState = game_state.copy()
                newGameState.buildCity(i)
                score = self.interpretState(newGameState)
                possbleActions.append(((3, i), score))
                    
        if actionMask[4]:
            tradeMask:np.array = game_state.getTradeWithBankMask(game_state.currentPlayer)
            for i in range(5):
                if not tradeMask[i]:
                    continue
                for j in range(5):
                    if i == j:
                        continue
                    # calculate score
                    newGameState = game_state.copy()
                    newGameState.tradeWithBank(i, j)
                    score = self.interpretState(newGameState)
                    possbleActions.append(((4, (i, j)), score))
                    
        return possbleActions
    
    def getBestAction(self, game_state: CatanState):
        possbleActions = self.getPossibleActions(game_state)
        
        # sort possbleActions by score
        possbleActions.sort(key=lambda x: x[1], reverse=True)
        
        # take the best action
        return possbleActions[0][0]
        
    def make_move(self, game_state):
        possbleActions = self.getPossibleActions(game_state)
        
        # sort possbleActions by score
        possbleActions.sort(key=lambda x: x[1], reverse=True)
        
        # take the best action
        action = possbleActions[0][0]
        if action[0] == 0:
            print(" End turn")
            return game_state.endTurn()
        elif action[0] == 1:
            print(" Build road")
            return game_state.buildRoad(action[1])
        elif action[0] == 2:
            print(" Build settlement")
            return game_state.buildSettlement(action[1])
        elif action[0] == 3:
            print(" Build city")
            return game_state.buildCity(action[1])
        elif action[0] == 4:
            print(" Trade")
            return game_state.tradeWithBank(action[1][0], action[1][1])   
        
    
    def make_opening_move(self, game_state):
        # build a settlement
        possibleActions = []
        
        cornerMask:np.array = game_state.getCornerMask(game_state.currentPlayer)
        for i in range(54):
            if not cornerMask[i]:
                continue
            # calculate score
            newGameState = game_state.copy()
            newGameState.buildSettlement(i)
            score = self.interpretState(newGameState)
            possibleActions.append((i, score))
            
        possibleActions.sort(key=lambda x: x[1], reverse=True)

        game_state.buildSettlement(possibleActions[0][0])
        
        # build a road 
        possibleActions = []
        
        edgeMask:np.array = game_state.getEdgeMask(game_state.currentPlayer)
        for i in range(72):
            if edgeMask[i]:
                # calculate score
                newGameState = game_state.copy()
                newGameState.buildRoad(i)
                score = self.interpretState(newGameState)
                possibleActions.append((i, score))
                
        possibleActions.sort(key=lambda x: x[1], reverse=True)
        game_state.buildRoad(possibleActions[0][0])
        
        

winnersPoints = []
totalActions = []
totalRounds = []

def plot_durations(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Tree simulating...')
    plt.xlabel('Episode')
    plt.ylabel('winnersPoints')
    plt.plot(winnersPoints)
    plt.plot(totalActions)
    plt.plot(totalRounds)

    plt.pause(0.001)  # pause a bit so that plots are updated
     
if __name__ == "__main__":
    plt.ion()
    bot = firstTreeBot()

    gameCount = 0
    while True:
        gameCount += 1
        game = CatanState()
        while game.round < 2:
            bot.make_opening_move(game)
            game.endTurn()
        
        for i in range(1000):
            bot.make_move(game)
            # print(game.currentPlayer, game.round)
            # print(game.players[game.currentPlayer])
            if game.hasGameEnded():
                #logging.debug(f"Game ended, points: {game.points}, winner: {np.argmax(game.points)}, round: {game.round}, actions: {i}, actions/round: {i/game.round}")
                winnersPoints.append(game.points[np.argmax(game.points)])
                totalActions.append(i/100)
                totalRounds.append(game.round/10)
                plot_durations()
                
                break
        else:
            logging.debug(f"Game did not end in 1000 rounds, points: {game.points}, game: {gameCount}")
            winnersPoints.append(game.points[np.argmax(game.points)])
            totalActions.append(10)
            totalRounds.append(game.round/10)
            plot_durations()
            
            
