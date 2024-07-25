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

byte defaultPlayerLocation = 0;

void setupLeds(catanState *pstate) {
    FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
    FastLED.setBrightness(50);
    statePointer = pstate;
    setupPlayerLocationsLeds();
}

void setupPlayerLocationsLeds(){
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


    for (byte i = 16; i < NUM_LEDS; i++) {
        playerLocations[i] = &defaultPlayerLocation;
    }
}

void showPlayerLocations() {
    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = playerColors[*playerLocations[i]];
    }
    FastLED.show();
}