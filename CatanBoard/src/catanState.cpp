#include "catanState.h"

catanState::catanState() {
    for (int i = 0; i < 19; i++) {
        hexes[i] = hex();
    }
    for (int i = 0; i < 54; i++) {
        corners[i] = corner();
        Serial.println("Corner " + String(i) + " created: " + String(corners[i].player));
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

    // check if the hex is valid
    if (hexY < 0 || hexY > 4) {
        return nullptr;
    }

    if (hexY <= 2){
        if (hexX < 0 || hexX > 2 + hexY) {
            return nullptr;
        }
    } else {
        if (hexX < hexY - 2 || hexX > 4) {
            return nullptr;
        }
    }

    return &hexes[hexIndex];
}

byte* catanState::getHexLocation(hex* hex) {
    byte hexIndex = 0;
    const byte rowStartIndex[5] = {0, 3, 7, 11, 14};
    for (int i = 0; i < 19; i++) {
        if (&hexes[i] == hex) {
            hexIndex = i;
            break;
        }
    }
    byte hexLocation[2];
    for (int i = 0; i < 5; i++) {
        if (hexIndex >= rowStartIndex[i] && hexIndex < rowStartIndex[i+1]) {
            hexLocation[0] = hexIndex - rowStartIndex[i];
            hexLocation[1] = i;
            break;
        }
    }
    return hexLocation;
}

hex** catanState::getHexesFromCorner(byte cornerX, byte cornerY) {
    hex** hexes = new hex*[3];
    // Get the hexes that are adjacent to the corner
    // (2,1) -> (0,0) (0,1) (1,1)
    // (2,2) -> (0,1) (0,2) (1,2)

    // (3,1) -> (0,0) (1,0) (1,1)
    // (3,2) -> (0,1) (1,1) (1,2)
    // (3,3) -> (0,2) (1,2) (1,3)

    // (4,1) -> (1,0) (1,1) (2,1)
    // (4,2) -> (1,1) (1,2) (2,2)
    // (4,3) -> (1,2) (1,3) (2,3)

    // (5,1) -> (1,0) (2,0) (2,1)
    // (5,2) -> (1,1) (2,1) (2,2)
    // (5,3) -> (1,2) (2,2) (2,3)   

    // (6,1) -> (2,0) (2,1) (3,1)
    // (6,2) -> (2,1) (2,2) (3,2)
    // (6,3) -> (2,2) (2,3) (3,3)

    if (cornerX % 2 == 0){
        hexes[0] = getHex((cornerX-2)/2, cornerY-1);
        hexes[1] = getHex((cornerX-2)/2, cornerY);
        hexes[2] = getHex((cornerX-2)/2 + 1, cornerY);
    }else{
        hexes[0] = getHex((cornerX-3)/2, cornerY-1);
        hexes[1] = getHex((cornerX-3)/2 + 1, cornerY-1);
        hexes[2] = getHex((cornerX-3)/2 + 1, cornerY);
    }
    return hexes;
}

corner* catanState::getCorner(byte cornerX, byte cornerY) {
    const byte rowStartIndex[6] = {0, 7, 16, 26, 35, 42};
    byte cornerIndex = 0;
    cornerIndex += rowStartIndex[cornerY] + cornerX;

    // check if the corner is valid
    if (cornerY < 0 || cornerY > 5) {
        return nullptr;
    }

    if (cornerY <= 2){
        if (cornerX < 0 || cornerX > 6 + cornerY * 2) {
            return nullptr;
        }
    } else {
        if (cornerX < 2 * (cornerY - 2) - 1 || cornerX > 3) {
            return nullptr;
        }
    }

    return &corners[cornerIndex];
}

byte* catanState::getCornerLocation(corner* corner) {
    byte cornerIndex = 0;
    const byte rowStartIndex[6] = {0, 7, 16, 26, 35, 42};
    for (int i = 0; i < 54; i++) {
        if (&corners[i] == corner) {
            cornerIndex = i;
            break;
        }
    }
    byte* cornerLocation = new byte[2];
    for (int i = 0; i < 6; i++) {
        if (cornerIndex >= rowStartIndex[i] && cornerIndex < rowStartIndex[i+1]) {
            cornerLocation[0] = cornerIndex - rowStartIndex[i];
            cornerLocation[1] = i;
            break;
        }
    }
    return cornerLocation;
}

corner** catanState::getCornersFromHex(byte hexX, byte hexY) {
    corner** corners = new corner*[6];

    corners[0] = getCorner(hexX*2, hexY);
    corners[1] = getCorner(hexX*2 + 1, hexY);
    corners[2] = getCorner(hexX*2 + 2, hexY);
    corners[3] = getCorner(hexX*2 + 1, hexY + 1);
    corners[4] = getCorner(hexX*2 + 2, hexY + 1);
    corners[5] = getCorner(hexX*2 + 3, hexY + 1);

    return corners;
}

corner** catanState::getSurroundingCorners(byte cornerX, byte cornerY) {
    corner** corners = new corner*[3];

    // Get the corners that are adjacent to the corner
    // (2,1) -> (1,1) (3,2) (3,1)
    // (2,2) -> (1,2) (3,3) (3,2)

    // (3,1) -> (2,0) (2,1) (4,1)
    // (3,2) -> (2,1) (2,2) (4,2)
    // (3,3) -> (2,2) (2,3) (4,3)

    // (4,1) -> (3,1) (5,2) (5,1)
    // (4,2) -> (3,2) (5,3) (5,2)

    if (cornerX % 2 == 0){
        corners[0] = getCorner(cornerX - 1, cornerY);
        corners[1] = getCorner(cornerX + 1, cornerY + 1);
        corners[2] = getCorner(cornerX + 1, cornerY);
    }else{
        corners[0] = getCorner(cornerX - 1, cornerY - 1);
        corners[1] = getCorner(cornerX - 1, cornerY);
        corners[2] = getCorner(cornerX + 1, cornerY);
    }
    
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
    type = playerType::NOPLAYER;
    resources[resourceType::BRICK] = 0;
    resources[resourceType::WOOD] = 0;
    resources[resourceType::SHEEP] = 0;
    resources[resourceType::WHEAT] = 0;
    resources[resourceType::ORE] = 0;
}