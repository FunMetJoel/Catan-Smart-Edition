from __future__ import annotations

class HexCoordinate:
    """
    Coordinate system for hex tiles.
    """

    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

    def __add__(self, other:HexCoordinate):
        return HexCoordinate(self.x + other.x, self.y + other.y)
    
    def __eq__(self, value):
        return self.x == value.x and self.y == value.y

    def __str__(self):
        return f"Hex({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def neighbors(self):
        """
        Returns a list of neighboring coordinates.
        """
        directions = [
            HexCoordinate(1, 0),
            HexCoordinate(1, 1),
            HexCoordinate(0, 1),
            HexCoordinate(-1, 0),
            HexCoordinate(-1, -1),
            HexCoordinate(0, -1)
        ]
        return [self + direction for direction in directions]
    
    def corners(self):
        """
        Returns a list of corners.
        """
        cordinates = [
            CornerCoordinate(self.x*2, self.y),
            CornerCoordinate(self.x*2+1, self.y),
            CornerCoordinate(self.x*2+2, self.y),

            CornerCoordinate(self.x*2+1, self.y + 1),
            CornerCoordinate(self.x*2+2, self.y + 1),
            CornerCoordinate(self.x*2+3, self.y + 1),
        ]
        return cordinates
    

class CornerCoordinate:
    """
    Coordinate system for corners of hex tiles.
    """

    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

    def __add__(self, other:CornerCoordinate):
        return CornerCoordinate(self.x + other.x, self.y + other.y)
    
    def __eq__(self, value):
        return self.x == value.x and self.y == value.y

    def __str__(self):
        return f"Corner({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def neighbors(self):
        """
        Returns a list of neighboring coordinates.
        """
        if self.x % 2 == 0:
            directions = [
                CornerCoordinate(1,1),
                CornerCoordinate(1,0),
                CornerCoordinate(-1,0)
            ]
        else:
            directions = [
                CornerCoordinate(-1,-1),
                CornerCoordinate(1,0),
                CornerCoordinate(-1,0)
            ]
        return [self + direction for direction in directions]
    
    def hexes(self):
        """
        Returns a list of hex coordinates.
        """
        if self.x % 2 == 0:
            cordinates = [
                HexCoordinate((self.x/2)-1, self.y),
                HexCoordinate(self.x/2, self.y),
                HexCoordinate((self.x/2)-1, self.y-1),
            ]
        else:
            cordinates = [
                HexCoordinate(((self.x-1)/2)-1, self.y-1),
                HexCoordinate((self.x-1)/2, self.y-1),
                HexCoordinate((self.x-1)/2, self.y),
            ]
        return cordinates
    

class EdgeCoordinate:
    """
    Coordinate system for edges of hex tiles.
    """

    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y

    def __add__(self, other:EdgeCoordinate):
        return EdgeCoordinate(self.x + other.x, self.y + other.y)
    
    def __eq__(self, value):
        return self.x == value.x and self.y == value.y

    def __str__(self):
        return f"Edge({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def neighbors(self):
        """
        Returns a list of neighboring coordinates.
        """
        if (self.y % 2 == 0) and (self.x % 2 == 0):
            directions = [
                EdgeCoordinate(1,0),
                EdgeCoordinate(-1,0),
                EdgeCoordinate(0,1),
                EdgeCoordinate(0,-1),
            ]
        elif (self.y % 2 == 0) and (self.x % 2 != 0):
            directions = [
                EdgeCoordinate(1,0),
                EdgeCoordinate(-1,0),
                EdgeCoordinate(1,1),
                EdgeCoordinate(-1,-1),
            ]
        elif (self.y % 2 != 0) and (self.x % 2 == 0):
            directions = [
                EdgeCoordinate(1,1),
                EdgeCoordinate(-1,-1),
                EdgeCoordinate(0,1),
                EdgeCoordinate(0,-1),
            ]
        return [self + direction for direction in directions]
    
    def hexes(self):
        """
        Returns a list of hex coordinates.
        """
        if (self.y % 2 == 0) and (self.x % 2 == 0):
            cordinates = [
                HexCoordinate((self.x/2)-1, (self.y/2)-1),
                HexCoordinate((self.x/2), (self.y/2)),
            ]
        elif (self.y % 2 == 0) and (self.x % 2 != 0):
            cordinates = [
                HexCoordinate((self.x-1)/2, (self.y/2)-1),
                HexCoordinate((self.x-1)/2, (self.y/2)),
            ]
        elif (self.y % 2 != 0) and (self.x % 2 == 0):
            cordinates = [
                HexCoordinate((self.x/2)-1, (self.y-1)/2),
                HexCoordinate((self.x/2), (self.y-1)/2),
            ]
        return cordinates