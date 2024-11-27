from __future__ import annotations
import enum
import random

class CatanState:
    def __init__(self, board:CatanBoard, players:list[CatanPlayer], current_player_id:int):
        self.board = board
        self.players = players
        self.current_player_id = current_player_id

    def __init__(self):
        self.board = CatanBoard()
        self.players = []
        self.current_player = 0

    def __str__(self):
        return f"Board: {self.board}\nPlayers: {self.players}\nCurrent Player: {self.current_player}\nDice Roll: {self.dice_roll}"
    
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
        rowsLengths = (5, 4, 7, 5, 9, 6, 9, 5, 7, 4, 5)
        xoffset = (0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= rowsLengths[y]:
            return None
        
        # x/(1 + (y % 2)) is the n'th road in the row. the (y % 2) is to account for the nonexistent roads in the odd rows
        return self.roads[sum(rowsLengths[:y]) + (x/(1 + (y % 2))) - xoffset[y]]
    
    def settlement(x, y) -> CatanSettlement:
        return


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
    def __init__(self, player):
        self.player = player

class CatanPlayer:
    def __init__(self, name, color, resources, settlements, cities, roads):
        self.name = name
        self.color = color
        self.resources = resources
        self.settlements = settlements
        self.cities = cities
        self.roads = roads