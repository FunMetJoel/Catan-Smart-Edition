#include "catanLeds.h"
#include <Arduino.h>
#include <FastLED.h>
#include "catanState.h"

CRGB playerColors[7] = {
    CRGB::Black,
    CRGB::Red,
    CRGB::Blue,
    CRGB::White,
    CRGB::Yellow,
    CRGB::Green,
    CRGB::Brown
};

#define NUM_LEDS 100
#define LED_PIN 6

CRGB leds[NUM_LEDS];

catanState *statePointer;
byte* playerLocations[NUM_LEDS];

byte defaultPlayerLocation = 2;

void setupLeds(catanState *pstate) {
    FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
    FastLED.setBrightness(50);
    statePointer = pstate;
    setupPlayerLocationsLeds();
}

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


}

void showPlayerLocations() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = playerColors[*playerLocations[i]];
    }
    FastLED.show();
}