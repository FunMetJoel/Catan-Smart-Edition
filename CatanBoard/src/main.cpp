#include <Arduino.h>
#include "catanLeds.h"
#include "catanState.h"

catanState state;

void setup() {
  state = catanState();
  setupLeds(&state);

  Serial.begin(9600);
}

void loop() {
  showPlayerLocations();

  // Read serial input until a newline character is received
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    Serial.println(input);

    // Parse the input
    String baseCommand = input.substring(0, input.indexOf(' '));
    String arguments = input.substring(input.indexOf(' ') + 1);
    if (baseCommand == "Corner"){
      byte cornerX = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte cornerY = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte player = arguments.toInt();
      state.getCorner(cornerX, cornerY)->player = player;
      Serial.println("Corner(" + String(cornerX) + ", " + String(cornerY) + ") set to " + String(player));
    } else if (baseCommand == "Edge"){
      byte edgeX = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte edgeY = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte player = arguments.toInt();
      state.getEdge(edgeX, edgeY)->player = player;
      Serial.println("Edge(" + String(edgeX) + ", " + String(edgeY) + ") set to " + String(player));
    }
  }
}
