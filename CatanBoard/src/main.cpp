#include <Arduino.h>
#include "catanLeds.h"
#include "catanState.h"

catanState state;

void setup() {
  setupLeds();
  state = catanState();
}

byte player = 1;

void loop() {
  showPlayerLocations(player);
  delay(1000);
  player++;
  if (player > 6) {
    player = 1;
  }
}
