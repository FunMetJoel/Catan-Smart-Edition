#include <Arduino.h>
#include "catanLeds.h"
#include "catanState.h"

catanState state;

void setup() {
  state = catanState();
  setupLeds(&state);

  Serial.begin(9600);


}

void setSurroundingCorners(byte cornerX, byte cornerY, byte p){
  corner** corners = state.getSurroundingCorners(cornerX, cornerY);
  for (byte i = 0; i < 3; i++){
    if (corners[i] == nullptr){
      Serial.println("Corner is null");
      continue;
    }
    // if (corners[i]->player > 0){
    //   Serial.println("Corner is already taken: " + String(corners[i]->player));
    //   continue;
    // }
    if (p > 2){
      Serial.println("Player is too high");
      continue;
    }
    Serial.println("Setting corner" + String(corners[i]->player) + " to " + String(p));
    corners[i]->player = p;
    byte* cornerLocation = state.getCornerLocation(corners[i]);
    Serial.println("Setting surrounding corners of " + String(cornerLocation[0]) + ", " + String(cornerLocation[1]));
    //setSurroundingCorners(cornerLocation[0], cornerLocation[1], p + 1);
  }
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
    } else if (baseCommand == "ShowC"){
      byte hexX = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte hexY = arguments.toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte player = arguments.toInt();
      corner** corners = state.getCornersFromHex(hexX, hexY);
      for (byte i = 0; i < 6; i++){
        corners[i]->player = player;
      }
    } else if (baseCommand == "TEST"){
      byte cornerX = arguments.substring(0, arguments.indexOf(' ')).toInt();
      arguments = arguments.substring(arguments.indexOf(' ') + 1);
      byte cornerY = arguments.toInt();
      Serial.println("TEST(" + String(cornerX) + ", " + String(cornerY) + ")");
      setSurroundingCorners(cornerX, cornerY, 1);
    }
  }
}


