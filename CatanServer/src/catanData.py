import numpy as np
import random
from compiledCordinateSystem import compiledHexIndex, compiledEdgeIndex, compiledCornerIndex, portCorners

class CatanState:
    def __init__(self):
        self.hexes = np.zeros((19, 3), dtype=int) # 19 hexes, 3 values per hex: resource, dice roll, hasRobber
        self.edges = np.zeros((72, 2), dtype=int) # 72 edges, 2 values per edge: player, hasRoad
        self.corners = np.zeros((54, 2), dtype=int) # 54 corners, 3 values per corner: player, level
        self.players = np.zeros((4, 5), dtype=int) # 4 players, 5 values per player: Wood, Brick, Wheat, Sheep, Ore
        self.points = np.zeros(4, dtype=int) # 4 players, 1 value per player: points
        self.ports = np.zeros(9, dtype=int) # 9 ports, 1 value per port: resource
        self.ontwikkelingskaarten = np.zeros((4, 5), dtype=int) # 5 values per player: ridder, overwinningpunt, monopolie, stratenbouwer, uitvinding
        
        self.currentPlayer = 0 # 0-3
        self.round = 0
        self.playerWithLongestRoad = -1
        self.lastRoll = 0
        
        self.canMoveRobber = False
        
        materials = [0, 0, 0, 0, # Wood
                     1, 1, 1, # Brick
                     2, 2, 2, 2, # Wheat
                     3, 3, 3, 3, # Sheep
                     4, 4, 4, # Ore
                     5 # Desert
                     ]
        
        # materials = 18 * [2] + [5]
        random.shuffle(materials)
        for i in range(19):
            self.hexes[i][0] = materials[i]
            
        diceRolls = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        random.shuffle(diceRolls)
        hasdonedesert = 0
        for i in range(19):
            if self.hexes[i][0] == 5:
                self.hexes[i][1] = 0 # dice roll
                self.hexes[i][2] = 1 # hasRobber
                hasdonedesert = 1
            else:
                self.hexes[i][1] = diceRolls[i - hasdonedesert]
                
        portOptions = [0, 1, 2, 3, 4, 5, 5, 5, 5]
        random.shuffle(portOptions)
        for i in range(9):
            self.ports[i] = portOptions[i]
            
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
        self.canMoveRobber = False
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
            
            # maby make it not be random
            # newRobberHex = np.random.randint(19)
            # for i in range(19):
            #     self.hexes[i][2] = 0
                
            #     if i == newRobberHex:
            #         self.hexes[i][2] = 1
            
            self.canMoveRobber = True
            
            
            for p in self.players:
                startAmount = sum(p)
                if startAmount <= 7:
                    continue
                while sum(p) > round(0.5*startAmount):
                    p[np.argmax(p)] -= 1
        
        for i in range(19):
            if self.hexes[i][1] == roll and self.hexes[i][2] == 0:
                for cornerIndex in compiledHexIndex.neighbourCorners[i]:
                    corner = self.corners[cornerIndex]
                    if corner[0] != 0:
                        self.players[corner[0]-1][self.hexes[i][0]] += corner[1]
                        
        self.lastRoll = roll
                        
        return roll
    
    def moveRobber(self, hexIndex):
        """moveRobberAction is called when the player moves the robber. It sets the hasRobber value of the hex.
        
        Args:
            hexIndex (int): The index of the hex to move the robber to.
        """
        for i in range(19):
            self.hexes[i][2] = 0
            if i == hexIndex:
                self.hexes[i][2] = 1
    
    def calculateLongestRoad(self, playerIndex) -> int:
        """calculateLongestRoad returns the length of the longest road for the player.
        
        Args:
            playerIndex (int): The index of the player.
        
        Returns:
            int: The length of the longest road for the player.
        """
        def dfs(roadIndex, visited: set) -> int:
            visited.add(roadIndex)
            max_length = 0
            for edgeIndex in compiledEdgeIndex.neighbourEdges[roadIndex]:
                if edgeIndex == -1:
                    continue
                if edgeIndex not in visited and self.edges[edgeIndex][0] == playerIndex+1:
                    max_length = max(max_length, dfs(edgeIndex, visited))
            visited.remove(roadIndex)
            return 1 + max_length

        max_route = 0
        for roadIndex in range(72):
            if self.edges[roadIndex][0] == playerIndex+1:
                max_route = max(max_route, dfs(roadIndex, set()))
                
        return max_route
        
    
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
            
        # calculate longest road
        longestRoad = self.calculateLongestRoad(self.currentPlayer)
        if longestRoad >= 5:
            if self.playerWithLongestRoad != -1:
                if longestRoad > self.calculateLongestRoad(self.playerWithLongestRoad):
                    self.points[self.playerWithLongestRoad] -= 2
                    self.playerWithLongestRoad = self.currentPlayer
                    self.points[self.currentPlayer] += 2
            else:
                self.playerWithLongestRoad = self.currentPlayer
                self.points[self.currentPlayer] += 2
        
        
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
                if hexIndex == -1 or self.hexes[hexIndex][0] == 5:
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
        
    def buyDevelopmentCard(self):
        """buyDevelopmentCardAction is called when the player buys a development card. It gives the player a random development card.
        """
        self.players[self.currentPlayer][2] -= 1
        self.players[self.currentPlayer][3] -= 1
        self.players[self.currentPlayer][4] -= 1
        
        posibleCards = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # ridder
            1, 1, 1, 1, 1, # overwinningpunt
            2, 2, # monopolie
            3, 3, # stratenbouwer
            4, 4 # uitvinding
        ]
        card = random.choice(posibleCards)
        
        self.players[self.currentPlayer][card] += 1
        
        
    def playDevelopmentCard(self, cardIndex, param):
        """playDevelopmentCardAction is called when the player plays a development card. It removes the card from the player.
        
        Args:
            cardIndex (int): The index of the card to play.
        """
        self.ontwikkelingskaarten[self.currentPlayer][cardIndex] -= 1
        
        if cardIndex == 0: # ridder
            self.canMoveRobber = True
        elif cardIndex == 1: # overwinningpunt
            self.points[self.currentPlayer] += 1
        elif cardIndex == 2: # monopolie
            for i in range(4):
                if i == self.currentPlayer:
                    continue
                self.players[self.currentPlayer][param] += self.players[i][param]
                self.players[i][param] = 0
        elif cardIndex == 3: # stratenbouwer
            self.players[self.currentPlayer][0] += 2
            self.players[self.currentPlayer][1] += 2
        elif cardIndex == 4: # uitvinding
            self.players[self.currentPlayer][param] += 2
            
    def getTradeRatios(self, playerIndex) -> np.array:
        """getTradeRatios returns the trade ratios of the player.
        
        Args:
            playerIndex (int): The index of the player.
        
        Returns:
            np.array: The trade ratios of the player.
        """
        ratios = np.zeros(5, dtype=int)
        
        ratios[0] = 4
        ratios[1] = 4
        ratios[2] = 4
        ratios[3] = 4
        ratios[4] = 4
        
        for i in range(9):
            for j in portCorners[i]:
                if self.corners[j][0] == playerIndex+1:
                    if self.ports[i] == 5:
                        for k in range(5):
                            if ratios[k] > 3:
                                ratios[k] = 3
                    else:
                        ratios[self.ports[i]] = 2
        
        return ratios
        
    def tradeWithBank(self, give, receive):
        """tradeWithBankAction is called when the player trades with the bank. It exchanges the give materials for the receive materials.
        
        Args:
            give (int): The index of the material to give.
            receive (int): The index of the material to receive.
        """
        racios = self.getTradeRatios(self.currentPlayer)
        
        self.players[self.currentPlayer][give] -= racios[give]
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
        if mask[1]:
            # count the number of roads from the player and check if < 15
            mask[1] = sum(self.edges[:,0] == self.currentPlayer+1) < 15
        mask[2] = self.players[self.currentPlayer][0] > 0 and self.players[self.currentPlayer][1] > 0 and self.players[self.currentPlayer][2] > 0 and self.players[self.currentPlayer][3] > 0
        if mask[2]:
            mask[2] = self.getCornerMask().any()
        if mask[2]:
            # count the number of curoners where corner[i][0] == player and corner[i][1] == 1 and check if < 5 
            curnersum = 0
            for i in range(54):
                if self.corners[i][0] == self.currentPlayer+1 and self.corners[i][1] == 1:
                    curnersum += 1
            mask[2] = curnersum < 5
            
        mask[3] = self.players[self.currentPlayer][2] > 1 and self.players[self.currentPlayer][4] > 2
        if mask[3]:
            mask[3] = self.getCityMask().any()
        if mask[3]:
            # count the number of curoners where corner[i][0] == player and corner[i][1] == 2 and check if < 4
            curnersum = 0
            for i in range(54):
                if self.corners[i][0] == self.currentPlayer+1 and self.corners[i][1] == 2:
                    curnersum += 1
            mask[3] = curnersum < 4
       
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
                        if cornerIndex == -1:
                            continue
                        if self.corners[cornerIndex][0] == playerIndex+1:
                            mask[i] = True
                            break
                    
                    for edgeIndex in compiledEdgeIndex.neighbourEdges[i]:
                        if edgeIndex == -1:
                            continue
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
                        if cornerIndex == -1:
                            continue
                        mask[cornerIndex] = False
            
        else:
            mask = np.zeros(54, dtype=bool)
            
            for i in range(54):
                if self.corners[i][0] == 0:
                    for edgeIndex in compiledCornerIndex.neighbourEdges[i]:
                        if edgeIndex == -1:
                            continue
                        if self.edges[edgeIndex][0] == player+1:
                            mask[i] = True
                            break
                    
                    for cornerIndex in compiledCornerIndex.neighbourCorners[i]: 
                        if cornerIndex == -1:
                            continue   
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
        
        