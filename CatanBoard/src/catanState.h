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

class catanState {
    public:
        catanState();
        hex* getHex(byte hexX, byte hexY);
        corner* getCorner(byte cornerX, byte cornerY);
        edge* getEdge(byte edgeX, byte edgeY);
    private:
        hex hexes[19];
        corner corners[54];
        edge edges[72];
        player players[6];
};

class hex {
    public:
        hex();
        hex(resourceType resource, byte number, bool robber);
    private:
        resourceType resource;
        byte number;
        bool robber;
};

class edge {
    public:
        edge();
        edge(byte player);
    private:
        byte player;
};

class corner {
    public:
        corner();
        corner(byte player, byte level);
    private:
        byte player;
        byte level;
};

enum playerType : byte {
    NONE = 0,
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

#endif // CATANSTATE_H