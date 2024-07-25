#include "catanState.h"

catanState::catanState() {
    for (int i = 0; i < 19; i++) {
        hexes[i] = hex();
    }
    for (int i = 0; i < 54; i++) {
        corners[i] = corner();
    }
    for (int i = 0; i < 72; i++) {
        edges[i] = edge();
    }
    for (int i = 0; i < 6; i++) {
        players[i] = player();
    }
}

hex* catanState::getHex(byte hexX, byte hexY) {
    const byte rowStartIndex[5] = {0, 3, 7, 11, 14};
    byte hexIndex = rowStartIndex[hexY] + hexX;
    return &hexes[hexIndex];
}

corner* catanState::getCorner(byte cornerX, byte cornerY) {
    const byte rowStartIndex[6] = {0, 7, 16, 26, 35, 42};
    byte cornerIndex = 0;
    cornerIndex += rowStartIndex[cornerY] + cornerX;
    return &corners[cornerIndex];
}

edge* catanState::getEdge(byte edgeX, byte edgeY) {
    const byte edgesPerRow[11] = {0, 6, 10, 18, 23, 33, 38, 48, 51, 60, 61};
    byte edgeIndex = edgesPerRow[edgeY];
    if (edgeY % 2 == 0) {
        edgeIndex += edgeX;
    } else {
        edgeIndex += edgeX / 2;
    }
    return &edges[edgeIndex];
}

hex::hex() {
    resource = resourceType::NONE;
    number = 0;
    robber = false;
}
hex::hex(resourceType resource, byte number, bool robber) {
    this->resource = resource;
    this->number = number;
    this->robber = robber;
}

edge::edge() {
    player = 0;
}
edge::edge(byte player) {
    this->player = player;
}

corner::corner() {
    player = 0;
    level = 0;
}
corner::corner(byte player, byte level) {
    this->player = player;
    this->level = level;
}

player::player() {
    type = playerType::NONE;
    resources[resourceType::BRICK] = 0;
    resources[resourceType::WOOD] = 0;
    resources[resourceType::SHEEP] = 0;
    resources[resourceType::WHEAT] = 0;
    resources[resourceType::ORE] = 0;
}