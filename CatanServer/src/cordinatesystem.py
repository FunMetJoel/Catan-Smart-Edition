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
    
    def neighbors(self) -> list[HexCoordinate]:
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
    
    def corners(self) -> list[CornerCoordinate]:
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
    
    def edges(self) -> list[EdgeCoordinate]:
        """
        Returns a list of edges.
        """
        cordinates = [
            EdgeCoordinate(self.x*2, self.y*2),
            EdgeCoordinate(self.x*2+1, self.y*2),
            EdgeCoordinate(self.x*2, self.y*2+1),
            EdgeCoordinate(self.x*2+2, self.y*2+1),
            EdgeCoordinate(self.x*2+1, self.y*2+2),
            EdgeCoordinate(self.x*2+2, self.y*2+2),
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
    
    def roads(self):
        """
        Returns a list of road coordinates.
        """

        # (1,1) -> (0,1) (0,2) (1,2)
        # (3,1) -> (2,1) (2,2) (3,2)
        # (3,2) -> (2,3) (2,4) (3,4)

        # (2,1) -> (1,2) (2,2) (2,3)
        # (4,2) -> (3,4) (4,4) (4,5)
        # (8,2) -> (7,4) (8,4) (8,5)

        if self.x % 2 == 0:
            cordinates = [
                EdgeCoordinate(self.x-1, self.y*2),
                EdgeCoordinate(self.x, self.y*2),
                EdgeCoordinate(self.x, self.y*2+1)
            ]
        else:
            cordinates = [
                EdgeCoordinate(self.x-1, self.y*2-1),
                EdgeCoordinate(self.x-1, self.y*2),
                EdgeCoordinate(self.x, self.y*2)
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

    def corners(self):
        """
        Returns a list of corner coordinates.
        """

        if self.y % 2 == 0:
            cordinates = [
                CornerCoordinate(self.x, round(self.y/2)),
                CornerCoordinate(self.x+1, round(self.y/2))
            ]
        else:
            cordinates = [
                CornerCoordinate(self.x, round((self.y-1)/2)),
                CornerCoordinate(self.x+1, round(1+((self.y-1)/2)))
            ]

        return cordinates


# Misschien niet hardcoden
HexCords: tuple[HexCoordinate] = (
    HexCoordinate(0, 0),
    HexCoordinate(1, 0),
    HexCoordinate(2, 0),
    HexCoordinate(0, 1),
    HexCoordinate(1, 1),
    HexCoordinate(2, 1),
    HexCoordinate(3, 1),
    HexCoordinate(0, 2),
    HexCoordinate(1, 2),
    HexCoordinate(2, 2),
    HexCoordinate(3, 2),
    HexCoordinate(4, 2),
    HexCoordinate(1, 3),
    HexCoordinate(2, 3),
    HexCoordinate(3, 3),
    HexCoordinate(4, 3),
    HexCoordinate(2, 4),
    HexCoordinate(3, 4),
    HexCoordinate(4, 4)
)

# Misschien niet hardcoden
CornerCords: tuple[CornerCoordinate] = (
    CornerCoordinate(0, 0),
    CornerCoordinate(1, 0),
    CornerCoordinate(2, 0),
    CornerCoordinate(3, 0),
    CornerCoordinate(4, 0),
    CornerCoordinate(5, 0),
    CornerCoordinate(6, 0),
    CornerCoordinate(0, 1),
    CornerCoordinate(1, 1),
    CornerCoordinate(2, 1),
    CornerCoordinate(3, 1),
    CornerCoordinate(4, 1),
    CornerCoordinate(5, 1),
    CornerCoordinate(6, 1),
    CornerCoordinate(7, 1),
    CornerCoordinate(8, 1),
    CornerCoordinate(0, 2),
    CornerCoordinate(1, 2),
    CornerCoordinate(2, 2),
    CornerCoordinate(3, 2),
    CornerCoordinate(4, 2),
    CornerCoordinate(5, 2),
    CornerCoordinate(6, 2),
    CornerCoordinate(7, 2),
    CornerCoordinate(8, 2),
    CornerCoordinate(9, 2),
    CornerCoordinate(10, 2),
    CornerCoordinate(1, 3),
    CornerCoordinate(2, 3),
    CornerCoordinate(3, 3),
    CornerCoordinate(4, 3),
    CornerCoordinate(5, 3),
    CornerCoordinate(6, 3),
    CornerCoordinate(7, 3),
    CornerCoordinate(8, 3),
    CornerCoordinate(9, 3),
    CornerCoordinate(10, 3),
    CornerCoordinate(11, 3),
    CornerCoordinate(3, 4),
    CornerCoordinate(4, 4),
    CornerCoordinate(5, 4),
    CornerCoordinate(6, 4),
    CornerCoordinate(7, 4),
    CornerCoordinate(8, 4),
    CornerCoordinate(9, 4),
    CornerCoordinate(10, 4),
    CornerCoordinate(11, 4),
    CornerCoordinate(5, 5),
    CornerCoordinate(6, 5),
    CornerCoordinate(7, 5),
    CornerCoordinate(8, 5),
    CornerCoordinate(9, 5),
    CornerCoordinate(10, 5),
    CornerCoordinate(11, 5)
)

# Misschien niet hardcoden
RoadCords: tuple[EdgeCoordinate] = (
    EdgeCoordinate(0, 0),
    EdgeCoordinate(1, 0),
    EdgeCoordinate(2, 0),
    EdgeCoordinate(3, 0),
    EdgeCoordinate(4, 0),
    EdgeCoordinate(5, 0),
    EdgeCoordinate(0, 1),
    EdgeCoordinate(2, 1),
    EdgeCoordinate(4, 1),
    EdgeCoordinate(6, 1),
    EdgeCoordinate(0, 2),
    EdgeCoordinate(1, 2),
    EdgeCoordinate(2, 2),
    EdgeCoordinate(3, 2),
    EdgeCoordinate(4, 2),
    EdgeCoordinate(5, 2),
    EdgeCoordinate(6, 2),
    EdgeCoordinate(7, 2),
    EdgeCoordinate(0, 3),
    EdgeCoordinate(2, 3),
    EdgeCoordinate(4, 3),
    EdgeCoordinate(6, 3),
    EdgeCoordinate(8, 3),
    EdgeCoordinate(0, 4),
    EdgeCoordinate(1, 4),
    EdgeCoordinate(2, 4),
    EdgeCoordinate(3, 4),
    EdgeCoordinate(4, 4),
    EdgeCoordinate(5, 4),
    EdgeCoordinate(6, 4),
    EdgeCoordinate(7, 4),
    EdgeCoordinate(8, 4),
    EdgeCoordinate(9, 4),
    EdgeCoordinate(0, 5),
    EdgeCoordinate(2, 5),
    EdgeCoordinate(4, 5),
    EdgeCoordinate(6, 5),
    EdgeCoordinate(8, 5),
    EdgeCoordinate(10, 5),
    EdgeCoordinate(1, 6),
    EdgeCoordinate(2, 6),
    EdgeCoordinate(3, 6),
    EdgeCoordinate(4, 6),
    EdgeCoordinate(5, 6),
    EdgeCoordinate(6, 6),
    EdgeCoordinate(7, 6),
    EdgeCoordinate(8, 6),
    EdgeCoordinate(9, 6),
    EdgeCoordinate(10, 6),
    EdgeCoordinate(2, 7),
    EdgeCoordinate(4, 7),
    EdgeCoordinate(6, 7),
    EdgeCoordinate(8, 7),
    EdgeCoordinate(10, 7),
    EdgeCoordinate(3, 8),
    EdgeCoordinate(4, 8),
    EdgeCoordinate(5, 8),
    EdgeCoordinate(6, 8),
    EdgeCoordinate(7, 8),
    EdgeCoordinate(8, 8),
    EdgeCoordinate(9, 8),
    EdgeCoordinate(10, 8),
    EdgeCoordinate(4, 9),
    EdgeCoordinate(6, 9),
    EdgeCoordinate(8, 9),
    EdgeCoordinate(10, 9),
    EdgeCoordinate(5, 10),
    EdgeCoordinate(6, 10),
    EdgeCoordinate(7, 10),
    EdgeCoordinate(8, 10),
    EdgeCoordinate(9, 10),
    EdgeCoordinate(10, 10)
)