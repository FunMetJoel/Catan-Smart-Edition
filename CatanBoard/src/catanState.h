#ifndef CATANSTATE_H
#define CATANSTATE_H

#include <Arduino.h>

enum resourceType : byte {
    NONE = 0,
    BRICK = 1,
    WOOD = 2,
    SHEEP = 3,
    WHEAT = 4,
    ORE = 5,
    DESERT = 6
};

class hex {
    public:
        hex();
        hex(resourceType resource, byte number, bool robber);
        resourceType resource;
        byte number;
        bool robber;
    private:   
};

class edge {
    public:
        edge();
        edge(byte player);
        byte player;
        void setPlayer();
    private:
        
};

class corner {
    public:
        corner();
        corner(byte player, byte level);
        byte player;
        byte level;
        void setData(byte player, byte level);
    private:
        
};

enum playerType : byte {
    NOPLAYER = 0,
    REAL = 1,
    AI = 2,
    SIMPLEBOT = 3
};

class player {
    public:
        player();
    private:
        playerType type;
        byte resources[5];
        byte devCards[5];
};

class catanState {
    public:
        catanState();
        hex* getHex(byte hexX, byte hexY);
        hex* getHexesFromCorner(byte cornerX, byte cornerY, byte i);
        corner* getCorner(byte cornerX, byte cornerY);
        corner* getCornersFromHex(byte hexX, byte hexY, byte i);
        corner* getSurroundingCorners(byte cornerX, byte cornerY, byte i);
        edge* getEdge(byte edgeX, byte edgeY);
        edge* getEdgesFromCorner(byte cornerX, byte cornerY, byte i);

        byte* getHexLocation(hex* hex);
        byte* getCornerLocation(corner* corner);
        byte* getEdgeLocation(edge* edge);

        hex hexes[19];
        corner corners[54];
        edge edges[72];
        player players[6];
    private:
        
};

#endif // CATANSTATE_H