#ifndef CATANLEDS_H
#define CATANLEDS_H

#include <FastLED.h>

// Declare any global variables or functions here
extern CRGB playerColors[7];

void setupLeds();
void showPlayerLocations(byte player);

#endif // CATANLEDS_H