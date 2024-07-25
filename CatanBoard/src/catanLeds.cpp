#include "catanLeds.h"
#include <Arduino.h>
#include <FastLED.h>

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

void setupLeds() {
    FastLED.addLeds<NEOPIXEL, LED_PIN>(leds, NUM_LEDS);
    FastLED.setBrightness(50);
}

void showPlayerLocations(byte player) {
    for (int i = 0; i < 100; i++) {
        leds[i] = playerColors[player];
    }
    FastLED.show();
}