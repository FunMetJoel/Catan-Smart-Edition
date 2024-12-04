from __future__ import annotations
import enum
import enums
from enums import Resource
from cordinatesystem import HexCoordinate, CornerCoordinate
import random
import sys
import time

class CatanState:
    def __init__(self, board:CatanBoard, players:list[CatanPlayer], current_player_id:int):
        self.board = board
        self.players = players
        self.current_player = current_player_id

    def __init__(self):
        self.board = CatanBoard()
        self.players = [CatanPlayer() for i in range(4)]
        self.current_player = 0

    def __str__(self):
        return f"Board: {self.board}\nPlayers: {self.players}\nCurrent Player: {self.current_player}\nDice Roll: {self.dice_roll}"
    
    def getRoadAvailabilty(self, player_id:int=None) -> list[bool]:
        roadAvailability = []
        if player_id == None:
            player_id = self.current_player
        for i in range(72):
            if self.board.roads[i].player != 0:
                roadAvailability.append(False)
                continue

            neighbors = enums.RoadCords[i].neighbors()
            for neighbor in neighbors:
                neighborRoad = self.board.road(neighbor.x, neighbor.y)
                if neighborRoad == None:
                    continue
                
                if neighborRoad.player == player_id:
                    roadAvailability.append(1)
                    break
            else:
                neighborCorners = enums.RoadCords[i].corners()
                for corner in neighborCorners:
                    neighborSettlement = self.board.settlement(corner.x, corner.y)
                    if neighborSettlement == None:
                        continue
                    
                    if neighborSettlement.player == player_id:
                        roadAvailability.append(1)
                        break
                else:
                    roadAvailability.append(0)
        return roadAvailability
    
    def getSettlementAvailabilty(self, player_id:int=None) -> list[bool]:
        settlementAvailability = []
        if player_id == None:
            player_id = self.current_player
        for i in range(54):
            if self.board.settlements[i].player != 0:
                settlementAvailability.append(False)
                continue

            neighbors = enums.CornerCords[i].neighbors()
            for neighbor in neighbors:
                neighborSettlement = self.board.settlement(neighbor.x, neighbor.y)
                if neighborSettlement == None:
                    continue
                
                if neighborSettlement.player != 0:
                    settlementAvailability.append(0)
                    break
            else:
                neighborroads = enums.CornerCords[i].roads()
                for road in neighborroads:
                    neighborRoad = self.board.road(road.x, road.y)
                    if neighborRoad == None:
                        continue
                    
                    if neighborRoad.player == player_id:
                        settlementAvailability.append(1)
                        break
                else:
                    settlementAvailability.append(0)
        return settlementAvailability

    def getActionAvailability(self, player_id:int=None) -> list[bool]:
        if player_id == None:
            player_id = self.current_player
        
        actions = [False for i in range(len(enums.Action))]
        actions[enums.Action.END_TURN] = True
        actions[enums.Action.BUILD_SETTLEMENT] = self.players[player_id].hasResources([1, 1, 1, 1, 0])
        actions[enums.Action.BUILD_CITY] = self.players[player_id].hasResources([0, 0, 3, 2, 0])
        actions[enums.Action.BUILD_ROAD] = self.players[player_id].hasResources([1, 1, 0, 0, 0])
        
        return actions
    
    
    
class CatanBoard:
    def __init__(self, tiles:list[CatanTile], roads:list[CatanRoad], settlements:list[CatanSettlement], cities):
        self.tiles = tiles
        self.roads = roads
        self.settlements = settlements
        self.cities = cities

    def __init__(self):
        self.tiles = [CatanTile for i in range(19)]
        for i in range(19):
            resourceType = random.choice([CatanResource.WOOD, CatanResource.BRICK, CatanResource.SHEEP, CatanResource.WHEAT, CatanResource.ORE])
            self.tiles[i] = CatanTile(resourceType, i % 11 + 2)
        self.roads = [CatanRoad(0) for i in range(72)]
        self.settlements = [CatanSettlement(0, 0) for i in range(54)]

    def hex(self, x, y) -> CatanTile:
        # self.tiles is a list of CatanTile objects with length 19
        # 3      16  17  18              (2,4) (3,4) (4,4)
        # 4    12  13  14  15         (1,3) (2,3) (3,3) (4,3)
        # 5  07  08  09  10  11    (0,2) (1,2) (2,2) (3,2) (4,2)
        # 4    03  04  05  06         (0,1) (1,1) (2,1) (3,1)
        # 3      00  01  02              (0,0) (1,0) (2,0)
        # 0, 0 is leftbottom corner
        rowsLengths = (3, 4, 5, 4, 3)
        xoffset = (0, 0, 0, 1, 2)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= rowsLengths[y]:
            return None
                
        return self.tiles[sum(rowsLengths[:y]) + x - xoffset[y]]

    def road(self, x, y) -> CatanRoad:
        # self.roads is a list of CatanRoad objects
        rowsLengths = (6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6)
        xoffset = (0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= (rowsLengths[y]*(1+(y%2))):
            return None
        
        # x/(1 + (y % 2)) is the n'th road in the row. the (y % 2) is to account for the nonexistent roads in the odd rows
        index = round(sum(rowsLengths[:y]) + ((x-xoffset[y])/(1 + (y % 2))))
        return self.roads[index]
    
    def settlement(self, x, y) -> CatanSettlement:
        # self.settlements is a list of CatanSettlement objects
        rowsLengths = (7, 9, 11, 11, 9, 7)
        xoffset = (0, 0, 0, 1, 3, 5)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= rowsLengths[y]:
            return None
        
        return self.settlements[sum(rowsLengths[:y]) + x - xoffset[y]]

class CatanResource(enum.IntEnum):
    WOOD = 1
    BRICK = 2
    SHEEP = 3
    WHEAT = 4
    ORE = 5

class CatanTile:
    def __init__(self, tile_type, number):
        self.tile_type = tile_type
        self.number = number

    def to_dict(self):
        return {
            "tile_type": self.tile_type,
            "number": self.number
        }

class CatanRoad:
    def __init__(self, player):
        self.player = player

class CatanSettlement:
    def __init__(self, player, level):
        self.player = player
        self.level = level


class CatanPlayer:
    def __init__(self, name, color, resources, settlements, cities, roads):
        self.name = name
        self.color = color
        self.resources = resources

    def __init__(self):
        self.resources = [0, 0, 0, 0, 0]
        
    def hasResources(self, resources):
        for i, resource in enumerate(resources):
            if self.resources[i] < resource:
                return False
        return True