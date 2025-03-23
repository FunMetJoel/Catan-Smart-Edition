#include "catanLeds.h"
#include <Arduino.h>
#include <FastLED.h>
#include "catanState.h"
#include "https.h"
#include <string>

#define NUM_LEDS 200
#define LED_PIN 2

CRGB leds[NUM_LEDS];

catanState *statePointer;
byte* playerLocations[NUM_LEDS];
byte* level[NUM_LEDS];

byte defaultPlayerLocation = 2;

void setupLeds(catanState *pstate) {
    FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
    FastLED.setBrightness(50);
    statePointer = pstate;
    setupPlayerLocationsLeds();
}
//filler V ?
void setupPlayerLocationsLeds(){
    for (byte i = 0; i < NUM_LEDS; i++) {
        playerLocations[i] = &defaultPlayerLocation;
    }

    playerLocations[0] = &statePointer->getEdge(0,3)->player;
    playerLocations[1] = &statePointer->getCorner(0,1)->player;
    playerLocations[2] = &statePointer->getEdge(0,2)->player;
    playerLocations[3] = &statePointer->getCorner(1,1)->player;
    playerLocations[4] = &statePointer->getEdge(1,2)->player;
    playerLocations[5] = &statePointer->getCorner(2,1)->player;
    playerLocations[6] = &statePointer->getEdge(2,3)->player;
    playerLocations[7] = &statePointer->getEdge(2,2)->player;
    playerLocations[8] = &statePointer->getCorner(3,1)->player;
    playerLocations[9] = &statePointer->getEdge(3,2)->player;
    playerLocations[10] = &statePointer->getCorner(4,1)->player;
    playerLocations[11] = &statePointer->getEdge(4,3)->player;
    playerLocations[12] = &statePointer->getCorner(5,2)->player;
    playerLocations[13] = &statePointer->getEdge(4,4)->player;
    playerLocations[14] = &statePointer->getEdge(4,2)->player;
    playerLocations[15] = &statePointer->getCorner(5,1)->player;
    playerLocations[16] = &statePointer->getEdge(5,2)->player;
    playerLocations[17] = &statePointer->getCorner(6,1)->player;
    playerLocations[18] = &statePointer->getEdge(6,3)->player;
    playerLocations[19] = &statePointer->getCorner(7,2)->player;
    playerLocations[20] = &statePointer->getEdge(6,4)->player;
    playerLocations[21] = &statePointer->getCorner(6,2)->player;
    playerLocations[22] = &statePointer->getEdge(5,4)->player;
    playerLocations[23] = &statePointer->getEdge(0,1)->player;
    playerLocations[24] = &statePointer->getCorner(0,0)->player;
    playerLocations[25] = &statePointer->getEdge(0,0)->player;
    playerLocations[26] = &statePointer->getCorner(1,0)->player;
    playerLocations[27] = &statePointer->getEdge(1,0)->player;
    playerLocations[28] = &statePointer->getCorner(2,0)->player;
    playerLocations[29] = &statePointer->getEdge(2,1)->player;
    playerLocations[30] = &statePointer->getEdge(2,0)->player;
    playerLocations[31] = &statePointer->getCorner(3,0)->player;
    playerLocations[32] = &statePointer->getEdge(3,0)->player;
    playerLocations[33] = &statePointer->getCorner(4,0)->player;
    playerLocations[34] = &statePointer->getEdge(4,1)->player;
    playerLocations[35] = &statePointer->getEdge(4,0)->player;
    playerLocations[36] = &statePointer->getCorner(5,0)->player;
    playerLocations[37] = &statePointer->getEdge(5,0)->player;
    playerLocations[38] = &statePointer->getCorner(6,0)->player;
    playerLocations[39] = &statePointer->getEdge(6,1)->player;
    playerLocations[40] = &statePointer->getCorner(7,1)->player;
    playerLocations[41] = &statePointer->getEdge(6,2)->player;
    playerLocations[42] = &statePointer->getEdge(7,2)->player;
    playerLocations[43] = &statePointer->getCorner(8,2)->player;
    playerLocations[44] = &statePointer->getEdge(8,3)->player;
    playerLocations[45] = &statePointer->getCorner(9,2)->player;
    playerLocations[46] = &statePointer->getEdge(8,4)->player;
}
//filler V
void CoardinatesExplination() {
    for (byte i = 0; i < NUM_LEDS; i++) {
        leds[i] = CRGB::Black;
    }

    leds[1] = CRGB(255, 0, 0);
    leds[3] = CRGB(0, 255, 0);
    leds[5] = CRGB(0, 0, 255);
    leds[8] = CRGB(255, 255, 0);
    leds[10] = CRGB(255, 0, 255);
    leds[12] = CRGB(0, 255, 255);
    leds[15] = CRGB(0, 255, 255);
    leds[17] = CRGB(255, 255, 255);
    leds[19] = CRGB(255, 0, 0);
    leds[21] = CRGB(255, 255, 255);
    leds[24] = CRGB(255, 0, 0);
    leds[26] = CRGB(0, 255, 0);
    leds[28] = CRGB(0, 0, 255);
    leds[31] = CRGB(255, 255, 0);
    leds[33] = CRGB(255, 0, 255);
    leds[36] = CRGB(0, 255, 255);
    leds[38] = CRGB(255, 255, 255);
    leds[40] = CRGB(255, 0, 0);

    FastLED.show();
}

CRGB playerColors[7] = {
    CRGB::Black,
    CRGB::Red,
    CRGB::Blue,
    CRGB::White,
    CRGB::Yellow,
    CRGB::Green,
    CRGB::Brown
};

void showPlayerLocations() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = playerColors[*playerLocations[i]];
    }
    FastLED.show();
}

// leds[i] = playerlocations[i]level[i]
// playerLocations[46] = &statePointer->corner(1)->player;

void colordivider() {
    // Declare a 2D array with player colors
    CRGB PlayerColorsArray[4][2] = {
        {CRGB::Blue, CRGB::LightBlue},    // Player 1: Blue and LightBlue
        {CRGB::Green, CRGB::LightGreen},  // Player 2: Green and LightGreen
        {CRGB::Yellow, CRGB::Orange},     // Player 3: Yellow and Orange
        {CRGB::Red, CRGB::Purple}         // Player 4: Red and Purple
    };

    for (int i = 0; i < NUM_LEDS; i++) {
        int playerIndex = *playerLocations[i];  // Get the player index from playerlocations array
        int levelIndex = *level[i];             // Get the level index from the level array
        
        // Set the LED color based on the player and level
        leds[i] = PlayerColorsArray[playerIndex][levelIndex];
    }
}

void errorLight() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CRGB(0, 0, 255);
    }
    FastLED.show();
    delay(2000);
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CRGB(0, 0, 0);
    }
    FastLED.show();
}

void currentPlayerLight() {
    CRGB currentPlayerLightList[4] = {CRGB::Red, CRGB::Blue, CRGB::Green, CRGB::Yellow};
    int currentplayer = 1;
    if (currentplayer > 4) {
        winnerLight(); 
    }
    currentplayer =- 1;
    for (int i = 0; i < 4; i++) {
        
        if (i = currentplayer) {
            leds[NUM_LEDS + i] = currentPlayerLightList[i];
        }
        else {
            leds[NUM_LEDS + i] = CRGB::Black;
        }
    }
}

void winnerLight() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CRGB(0, 255, 0);
    }       
}