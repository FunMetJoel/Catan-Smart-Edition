#ifndef CATANLEDS_H
#define CATANLEDS_H

#include <FastLED.h>
#include "catanState.h"

// Declare any global variables or functions here
extern CRGB playerColors[7];

void setupLeds(catanState *pstate);
void setupPlayerLocationsLeds();
void showPlayerLocations();
void CoardinatesExplination();
void errorLight();

#endif // CATANLEDS_H