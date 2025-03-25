/*----------------------------------/please read below\-----------------------------------------------------------
| Copyright (C) 2025 'team joël' - All Rights Reserved                                                             |
| You may not use, distribute and modify this code unless stated.                                                  |
| Failuire to comply will result in a | Class A Action Lawssuit |                                                  |
| under the federal Copyright infringement program.                                                                |
| Any questions may be directed to : www.idontcare@bouwtyouropinion.gmail.com :                                    |
-----------------------------------------------------------------------------------------------------------------*/
#include <Arduino.h>
#include "catanLeds.h"
#include "catanState.h"
#include "https.h"


catanState state;
bool error = false;
String currentPlayer;

void setup() {
  Serial.begin(9600);
  state = catanState();
  setupLeds(&state);
  setupHttp();
}



void loop() {
  if (error == false) {
    getCurrentPlayer(currentPlayer);
    if(currentPlayer == "1") { //if player = 1 your turn = true;
      //readpins();  <- 
      //setuppininfo; <- send the stuff to the main
    }
    getHex(&state);
    getHouse(&state);
    getRoad(&state); 
    colordivider();  
  }
  else {
    errorLight();
    delay(1000);
    setup();
    error = false; 
  }
}




// void setSurroundingCorners(byte cornerX, byte cornerY, byte p){
//   for (byte i = 0; i < 3; i++){
//     corner* c = state.getSurroundingCorners(cornerX, cornerY, i);
//     if (c != nullptr){
//       c->player = p;
//     }
//   }
// }

// void loop() {
//   showPlayerLocations();
//   //ServerGet();  //<- server lists pullen 
//   //getHex();
//   // Read serial input until a newline character is received
//   if (Serial.available() > 0) {
//     String input = Serial.readStringUntil('\n');
//     Serial.println(input);

//     // Parse the input
//     String baseCommand = input.substring(0, input.indexOf(' '));
//     String arguments = input.substring(input.indexOf(' ') + 1);
//     if (baseCommand == "Corner"){
//       byte cornerX = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte cornerY = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte player = arguments.toInt();
//       state.getCorner(cornerX, cornerY)->player = player;
//       Serial.println("Corner(" + String(cornerX) + ", " + String(cornerY) + ") set to " + String(player));
//     } else if (baseCommand == "Edge"){
//       byte edgeX = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte edgeY = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte player = arguments.toInt();
//       state.getEdge(edgeX, edgeY)->player = player;
//       Serial.println("Edge(" + String(edgeX) + ", " + String(edgeY) + ") set to " + String(player));
//     } else if (baseCommand == "ShowC"){
//       byte hexX = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte hexY = arguments.toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte player = arguments.toInt();
//       for (byte i = 0; i < 6; i++){
//         corner* c = state.getCornersFromHex(hexX, hexY, i);
//         if (c != nullptr) c->player = player;
//       }
//     } else if (baseCommand == "TEST"){
//       byte cornerX = arguments.substring(0, arguments.indexOf(' ')).toInt();
//       arguments = arguments.substring(arguments.indexOf(' ') + 1);
//       byte cornerY = arguments.toInt();
//       Serial.println("TEST(" + String(cornerX) + ", " + String(cornerY) + ")");
//       setSurroundingCorners(cornerX, cornerY, 1);
//     }
//   }
// }
