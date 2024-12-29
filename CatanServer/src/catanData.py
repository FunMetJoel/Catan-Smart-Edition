import numpy as np
import random
from compiledCordinateSystem import compiledHexIndex, compiledEdgeIndex, compiledCornerIndex

class CatanState:
    def __init__(self):
        self.hexes = np.zeros((19, 3), dtype=int) # 19 hexes, 3 values per hex: resource, dice roll, hasRobber
        self.edges = np.zeros((72, 2), dtype=int) # 72 edges, 2 values per edge: player, hasRoad
        self.corners = np.zeros((54, 2), dtype=int) # 54 corners, 3 values per corner: player, level
        self.players = np.zeros((4, 5), dtype=int) # 4 players, 5 values per player: Wood, Brick, Wheat, Sheep, Ore
        self.points = np.zeros(4, dtype=int) # 4 players, 1 value per player: points
        
        self.currentPlayer = 0 # 0-3
        self.round = 0
        
        materials = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        random.shuffle(materials)
        for i in range(19):
            self.hexes[i][0] = materials[i]
            
        diceRolls = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12, 0]
        random.shuffle(diceRolls)
        for i in range(19):
            self.hexes[i][1] = diceRolls[i]
            
    def setupRandomly(self):
        for i in range(4):
            for j in range(2):
                cornerMask = self.getCornerMask(i)
                randomIndex = np.random.randint(52)
                while not cornerMask[randomIndex]:
                    randomIndex = np.random.randint(52)
                self.corners[randomIndex][0] = i+1
                self.corners[randomIndex][1] = 1
                
                edgeMask = self.getEdgeMask(i)
                ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
                while not edgeMask[ramdomRoadIndex]:
                    ramdomRoadIndex = random.choice(compiledCornerIndex.neighbourEdges[randomIndex])
                self.edges[ramdomRoadIndex][0] = i+1
                self.edges[ramdomRoadIndex][1] = 1
                
        self.round = 2
            
    def copy(self):
        """copy returns a copy of the current state.
        
        Returns:
            CatanState: A copy of the current state.
        """
        copy = CatanState()
        
        copy.hexes = self.hexes.copy()
        copy.edges = self.edges.copy()
        copy.corners = self.corners.copy()
        copy.players = self.players.copy()
        copy.points = self.points.copy()
        
        copy.currentPlayer = self.currentPlayer
        copy.round = self.round
        
        return copy
    
    def hasGameEnded(self) -> bool:
        """hasGameEnded returns True if the game has ended, otherwise False.
        
        Returns:
            bool: True if the game has ended, otherwise False.
        """
        for points in self.points:
            if points >= 10:
                return True
        
        return False
    
    def endTurn(self):
        """endTurnAction is called when the player ends their turn. It increments the currentPlayer and the round.
        """
        if self.round == 0:
            self.currentPlayer = (self.currentPlayer + 1) % 4
            if self.currentPlayer == 0:
                self.round += 1
                self.currentPlayer = 3
        elif self.round == 1:
            self.currentPlayer = (self.currentPlayer - 1) % 4
            if self.currentPlayer == 3:
                self.round += 1
                self.currentPlayer = 0
        else:
            self.currentPlayer = (self.currentPlayer + 1) % 4
            if self.currentPlayer == 0:
                self.round += 1
        
            self.rollDice()
            
    def rollDice(self) -> int:
        """rollDiceAction is called when the player rolls the dice. It returns the sum of two random numbers between 1 and 6.
        
        Returns:
            int: The sum of two random numbers between 1 and 6.
        """
        roll = random.randint(1, 6) + random.randint(1, 6)
        
        if roll == 7:
            for p in self.players:
                while sum(p) > 7:
                    p[np.argmax(p)] -= 1
        
        for i in range(19):
            if self.hexes[i][1] == roll and self.hexes[i][2] == 0:
                for cornerIndex in compiledHexIndex.neighbourCorners[i]:
                    corner = self.corners[cornerIndex]
                    if corner[0] != 0:
                        self.players[corner[0]-1][self.hexes[i][0]] += corner[1]
                        
        return roll
    
    def buildRoad(self, edgeIndex):
        """buildRoadAction is called when the player builds a road. It sets the player and hasRoad values of the edge.
        
        Args:
            edgeIndex (int): The index of the edge to build the road on.
        """
        self.edges[edgeIndex][0] = self.currentPlayer+1
        self.edges[edgeIndex][1] = 1
        
        if self.round >= 2:
            self.players[self.currentPlayer][0] -= 1
            self.players[self.currentPlayer][1] -= 1
        
        
    def buildSettlement(self, cornerIndex):
        """buildSettlementAction is called when the player builds a settlement. It sets the player and level values of the corner.
        
        Args:
            cornerIndex (int): The index of the corner to build the settlement on.
        """
        # print(cornerIndex)
        
        self.corners[cornerIndex][0] = self.currentPlayer+1
        self.corners[cornerIndex][1] = 1
        
        if self.round >= 2:
            self.players[self.currentPlayer][0] -= 1
            self.players[self.currentPlayer][1] -= 1
            self.players[self.currentPlayer][2] -= 1
            self.players[self.currentPlayer][3] -= 1
        else:
            # give player materials 
            for hexIndex in compiledCornerIndex.neighbourHexes[cornerIndex]:
                if hexIndex == -1:
                    continue
                self.players[self.currentPlayer][self.hexes[hexIndex][0]] += 1
        
        self.points[self.currentPlayer] += 1
        
    def buildCity(self, cornerIndex):
        """buildCityAction is called when the player builds a city. It sets the player and level values of the corner.
        
        Args:
            cornerIndex (int): The index of the corner to build the city on.
        """
        self.corners[cornerIndex][0] = self.currentPlayer+1
        self.corners[cornerIndex][1] = 2
        
        self.players[self.currentPlayer][2] -= 2
        self.players[self.currentPlayer][4] -= 3
        
        self.points[self.currentPlayer] += 1
        
    def tradeWithBank(self, give, receive):
        """tradeWithBankAction is called when the player trades with the bank. It exchanges the give materials for the receive materials.
        
        Args:
            give (int): The index of the material to give.
            receive (int): The index of the material to receive.
        """
        self.players[self.currentPlayer][give] -= 4
        self.players[self.currentPlayer][receive] += 1
    
    def getActionMask(self) -> np.array:
        """getActionMask returns a mask of the actions that the current player can take.
        
        Returns:
            np.array: A mask of the actions that the current player can take.
        """
        mask = np.zeros(5, dtype=bool)
        
        mask[0] = True
        mask[1] = self.players[self.currentPlayer][0] > 0 and self.players[self.currentPlayer][1] > 0
        if mask[1]:
            mask[1] = self.getEdgeMask().any()
        mask[2] = self.players[self.currentPlayer][0] > 0 and self.players[self.currentPlayer][1] > 0 and self.players[self.currentPlayer][2] > 0 and self.players[self.currentPlayer][3] > 0
        if mask[2]:
            mask[2] = self.getCornerMask().any()
        mask[3] = self.players[self.currentPlayer][2] > 1 and self.players[self.currentPlayer][4] > 2
        if mask[3]:
            mask[3] = self.getCityMask().any()
        mask[4] = any(self.players[self.currentPlayer] > 4)
        
        return mask
    
    def getEdgeMask(self, playerIndex=None) -> np.array:
        """getEdgeMask returns a mask of the edges that the current player can build on.
        
        Returns:
            np.array: A mask of the edges that the current player can build on.
        """
        if playerIndex is None:
            playerIndex = self.currentPlayer
        
        mask = np.zeros(72, dtype=bool)
        
        if self.round < 2:
            # loop trough all corners
            for i in range(54):
                if self.corners[i][0] != playerIndex+1:
                    continue
                
                # check if already has a road
                for edgeIndex in compiledCornerIndex.neighbourEdges[i]:
                    if self.edges[edgeIndex][0] == playerIndex+1:
                        break
                else:
                    # does not have a road
                    for edgeIndex in compiledCornerIndex.neighbourEdges[i]:
                        mask[edgeIndex] = True
                        
        else:
            for i in range(72):
                if self.edges[i][0] == 0:
                    # mask[i] = True
                    for cornerIndex in compiledEdgeIndex.neighbourCorners[i]:
                        if self.corners[cornerIndex][0] == playerIndex+1:
                            mask[i] = True
                            break
                    
                    for edgeIndex in compiledEdgeIndex.neighbourEdges[i]:
                        if self.edges[edgeIndex][0] == playerIndex+1:
                            mask[i] = True
                            break
                
        return mask
    
    def getCornerMask(self, player=None) -> np.array:
        """getCornerMask returns a mask of the corners that the current player can build on.
        
        Returns:
            np.array: A mask of the corners that the current player can build on.
        """
        if player is None:
            player = self.currentPlayer
            
        if self.round < 2:
            mask = np.ones(54, dtype=bool)
            
            for i in range(54):
                if self.corners[i][0] != 0:
                    mask[i] = False
                    for cornerIndex in compiledCornerIndex.neighbourCorners[i]:
                        mask[cornerIndex] = False
            
        else:
            mask = np.zeros(54, dtype=bool)
            
            for i in range(54):
                if self.corners[i][0] == 0:
                    for edgeIndex in compiledCornerIndex.neighbourEdges[i]:
                        if self.edges[edgeIndex][0] == player+1:
                            mask[i] = True
                            break
                    
                    for cornerIndex in compiledCornerIndex.neighbourCorners[i]:    
                        if self.corners[cornerIndex][0] != 0:
                            mask[i] = False
                            break
                    
        return mask
        
    def getCityMask(self, player=None) -> np.array:
        """getCityMask returns a mask of the corners that the current player can build a city on.
        
        Returns:
            np.array: A mask of the corners that the current player can build a city on.
        """
        if player is None:
            player = self.currentPlayer
        
        mask = np.zeros(54, dtype=bool)
        
        for i in range(54):
            if self.corners[i][0] == player+1 and self.corners[i][1] == 1:
                mask[i] = True
        
        return mask
    
    def getTradeWithBankMask(self, player=None) -> np.array:
        """getTradeWithBankMask returns a mask of the trades that the current player can make with the bank.
        
        Returns:
            np.array: A mask of the trades that the current player can make with the bank.
        """
        if player is None:
            player = self.currentPlayer
        
        mask = np.zeros(5, dtype=bool)
        
        mask[0] = self.players[player][0] > 3
        mask[1] = self.players[player][1] > 3
        mask[2] = self.players[player][2] > 3
        mask[3] = self.players[player][3] > 3
        mask[4] = self.players[player][4] > 3
        
        return mask
        
        