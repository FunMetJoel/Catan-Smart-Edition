import enum
from __future__ import annotations

class CatanState:
    def __init__(self, board:CatanBoard, players:list[CatanPlayer], current_player_id:int):
        self.board = board
        self.players = players
        self.current_player_id = current_player_id

    def __str__(self):
        return f"Board: {self.board}\nPlayers: {self.players}\nCurrent Player: {self.current_player}\nDice Roll: {self.dice_roll}"
    
class CatanBoard:
    def __init__(self, tiles:list[CatanTile], roads:list[CatanRoad], settlements:list[CatanSettlement], cities):
        self.tiles = tiles
        self.roads = roads
        self.settlements = settlements
        self.cities = cities

class CatanResource(enum.Enum, int):
    WOOD = 1
    BRICK = 2
    SHEEP = 3
    WHEAT = 4
    ORE = 5

class CatanTile:
    def __init__(self, tile_type, number):
        self.tile_type = tile_type
        self.number = number

class CatanRoad:
    def __init__(self, player):
        self.player = player
        self.hasRou

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