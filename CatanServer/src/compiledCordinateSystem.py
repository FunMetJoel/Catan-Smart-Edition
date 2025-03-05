import numpy as np
import cordinatesystem

class CompiledHexIndex:
    def __init__(self):
        self.neighbourHexes = np.zeros((19, 6), dtype=int) # 19 hexes, 6 values per hex: index of neighbour hexes, -1 if no neighbour
        self.neighbourEdges = np.zeros((19, 6), dtype=int) # 19 hexes, 6 values per hex: index of neighbour edges, -1 if no neighbour
        self.neighbourCorners = np.zeros((19, 6), dtype=int) # 19 hexes, 6 values per hex: index of neighbour corners, -1 if no neighbour

        for i in range(19):
            hexCordinate = cordinatesystem.HexCords[i]
            for j in range(6):
                neighbourHex = hexCordinate.neighbors()[j]
                neighbourEdge = hexCordinate.edges()[j]
                neighbourCorner = hexCordinate.corners()[j]
                
                neighbourHexIndex = self.gethexindex(neighbourHex.x, neighbourHex.y)
                if neighbourHexIndex is None:
                    self.neighbourHexes[i][j] = -1
                else:
                    self.neighbourHexes[i][j] = neighbourHexIndex
                
                neighbourEdgeIndex = CompiledEdgeIndex.calcEdgeIndex(neighbourEdge.x, neighbourEdge.y)
                if neighbourEdgeIndex is None:
                    self.neighbourEdges[i][j] = -1
                else:
                    self.neighbourEdges[i][j] = neighbourEdgeIndex
                    
                neighbourCornerIndex = CompiledCornerIndex.calcCornerIndex(neighbourCorner.x, neighbourCorner.y)
                if neighbourCornerIndex is None:
                    self.neighbourCorners[i][j] = -1
                else:
                    self.neighbourCorners[i][j] = neighbourCornerIndex
        
    @staticmethod
    def gethexindex(x:int, y:int):
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
        if x < xoffset[round(y)] or x-xoffset[round(y)] >= rowsLengths[round(y)]:
            return None
                
        return sum(rowsLengths[:round(y)]) + x - xoffset[round(y)]
        
class CompiledEdgeIndex:
    def __init__(self):
        self.neighbourEdges = np.zeros((72, 4), dtype=int) # 72 edges, 4 values per edge: index of neighbour edges, -1 if no neighbour
        self.neighbourHexes = np.zeros((72, 2), dtype=int) # 72 edges, 2 values per edge: index of neighbour hexes, -1 if no neighbour
        self.neighbourCorners = np.zeros((72, 2), dtype=int) # 72 edges, 2 values per edge: index of neighbour corners, -1 if no neighbour
        
        for i in range(72):
            edgeCordinate = cordinatesystem.RoadCords[i]
            for j in range(4):
                neighbourEdge = edgeCordinate.neighbors()[j]
                neighbourEdgeIndex = CompiledEdgeIndex.calcEdgeIndex(neighbourEdge.x, neighbourEdge.y)
                if neighbourEdgeIndex is None:
                    self.neighbourEdges[i][j] = -1
                else:
                    self.neighbourEdges[i][j] = neighbourEdgeIndex
                
            for j in range(2):
                neighbourHex = edgeCordinate.hexes()[j]
                neighbourCorner = edgeCordinate.corners()[j]
                
                neighbourHexIndex = CompiledHexIndex.gethexindex(neighbourHex.x, neighbourHex.y)
                if neighbourHexIndex is None:
                    self.neighbourHexes[i][j] = -1
                else:
                    self.neighbourHexes[i][j] = neighbourHexIndex
                    
                neighbourCornerIndex = CompiledCornerIndex.calcCornerIndex(neighbourCorner.x, neighbourCorner.y)
                if neighbourCornerIndex is None:
                    self.neighbourCorners[i][j] = -1
                else: 
                    self.neighbourCorners[i][j] = neighbourCornerIndex
        
    @staticmethod
    def calcEdgeIndex(x, y):
        # self.roads is a list of CatanRoad objects
        rowsLengths = (6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6)
        xoffset = (0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= (rowsLengths[y]*(1+(y%2))):
            return None
        
        # x/(1 + (y % 2)) is the n'th road in the row. the (y % 2) is to account for the nonexistent roads in the odd rows
        return round(sum(rowsLengths[:y]) + ((x-xoffset[y])/(1 + (y % 2))))
        
class CompiledCornerIndex:
    def __init__(self):
        self.neighbourCorners = np.zeros((54, 3), dtype=int) # 54 corners, 3 values per corner: index of neighbour corners, -1 if no neighbour
        self.neighbourEdges = np.zeros((54, 3), dtype=int) # 54 corners, 3 values per corner: index of neighbour edges, -1 if no neighbour
        self.neighbourHexes = np.zeros((54, 3), dtype=int) # 54 corners, 3 values per corner: index of neighbour hexes, -1 if no neighbour
        
        for i in range(54):
            cornerCordinate = cordinatesystem.CornerCords[i]
            for j in range(3):
                neighbourCorner = cornerCordinate.neighbors()[j]
                neighbourEdge = cornerCordinate.roads()[j]
                neighbourHex = cornerCordinate.hexes()[j]
                
                neighbourCornerIndex = CompiledCornerIndex.calcCornerIndex(neighbourCorner.x, neighbourCorner.y)
                if neighbourCornerIndex is None:
                    self.neighbourCorners[i][j] = -1
                else:
                    self.neighbourCorners[i][j] = neighbourCornerIndex
                    
                neighbourEdgeIndex = CompiledEdgeIndex.calcEdgeIndex(neighbourEdge.x, neighbourEdge.y)
                if neighbourEdgeIndex is None:
                    self.neighbourEdges[i][j] = -1
                else:
                    self.neighbourEdges[i][j] = neighbourEdgeIndex
                    
                neighbourHexIndex = CompiledHexIndex.gethexindex(neighbourHex.x, neighbourHex.y)
                if neighbourHexIndex is None:
                    self.neighbourHexes[i][j] = -1
                else:
                    self.neighbourHexes[i][j] = neighbourHexIndex
        
    @staticmethod
    def calcCornerIndex(x, y):
        rowsLengths = (7, 9, 11, 11, 9, 7)
        xoffset = (0, 0, 0, 1, 3, 5)
        if y < 0 or y >= len(rowsLengths):
            return None
        if x < xoffset[y] or x-xoffset[y] >= rowsLengths[y]:
            return None
        
        return sum(rowsLengths[:y]) + x - xoffset[y]

compiledHexIndex:CompiledHexIndex = CompiledHexIndex()
compiledEdgeIndex = CompiledEdgeIndex()
compiledCornerIndex = CompiledCornerIndex()

# what corners each port is connected to
portCorners = [
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(0, 0)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(3, 0)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(7, 2)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(0, 3)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(10, 5)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(2, 7)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(10, 8)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(5, 10)],
    compiledEdgeIndex.neighbourCorners[compiledEdgeIndex.calcEdgeIndex(8, 10)]
]