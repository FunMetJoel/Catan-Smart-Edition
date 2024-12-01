import enum
from cordinatesystem import HexCoordinate, EdgeCoordinate, CornerCoordinate

class Resource(enum.IntEnum):
    WOOD = 1
    BRICK = 2
    SHEEP = 3
    WHEAT = 4
    ORE = 5

class TileType(enum.IntEnum):
    DESERT = 0
    WOOD = 1
    BRICK = 2
    SHEEP = 3
    WHEAT = 4
    ORE = 5

class DevelopmentCard(enum.IntEnum):
    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    MONOPOLY = 3
    YEAR_OF_PLENTY = 4

class Action(enum.IntEnum):
    END_TURN = 0
    BUILD_SETTLEMENT = 1
    BUILD_CITY = 2
    BUILD_ROAD = 3

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