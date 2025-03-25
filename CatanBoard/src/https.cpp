#include "https.h"
#include <Arduino.h>
#include "catanState.h"
#include <WiFi.h>
#include <HTTPClient.h>

bool error = false;
const char* ssid = "Klinki";
const char* password = "KlinkiSnek";
catanState *statePointer;

String serverRoad = "http://192.168.1.35:5000";
String serverHouse = "http://192.168.1.35:5000"; //even van jowel vragen
String serverHex = "http://192.168.1.35:5000";
String serverCity = "http://192.168.1.35:5000";



void setupHttp() {
  statePointer = pstate;
  WiFi.begin(ssid, password);
  if(WiFi.status() != WL_CONNECTED) {
    bool wificonnect = true; //<-deze kan weg

  }   
  else {
    error = true;
  }
}

void getRoad(catanState *pstate) {
  HTTPClient http;
  http.begin(serverRoad.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String roadData = http.getString();
    for (int i = 0; i < 72; i++){
      &statePointer->edges[i].setPlayer() = roadData[i];
    }
  }
  else {
    error = true;
  }
}

void getHouse(catanState *pstate) {
  HTTPClient http;
  http.begin(serverHouse.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String houseData = http.getString();
    for (int i = 0; i < 54; i++){
      &statePointer->corners[i].player = houseData[i * 2];
      &statePointer->corners[i].level = houseData[i * 2 + 1];
    }
  }
  else {
    error = true;
  }
}

void getHex(catanState *pstate) {
  HTTPClient http;
  http.begin(serverHex.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String hexData = http.getString();
    for (int i = 0; i < 72; i++){
      &statePointer->hexes[i].resource = hexData[i * 2];
      &statePointer->hexes[i].robber = hexData[i * 2 + 1];
    }
  }
 else {
  error = true;
  }
}

void putRoad() {
//get httppath = url/welke road bouwen serverHex + "/71"
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverHexPutPath = serverHex.c_str(); //+  //number; //<- 
    http.begin(serverHexPutPath.c_str());
    int httpResponseCode = http.GET();
  }
}

void putHouse() {
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverHexPutPath = serverHouse.c_str(); //+  //number; //<- 
    http.begin(serverHexPutPath.c_str());
    int httpResponseCode = http.GET();
  }
  //response code get?
}

void putCity() {
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverCityPutPath = serverCity.c_str(); //+  //number; //<- 
    http.begin(serverCityPutPath.c_str());
    int httpResponseCode = http.GET();
  }
}

void getCurrentPlayer(String currentPlayer) {
  HTTPClient http;
  http.begin(serverHex.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String currentPlayerString = http.getString();  // Get the currentPlayer as a string
    currentPlayer = currentPlayerString.toInt();
  }
  // current player = welke aan de beurt is 
}


/*ToDo--------------
- put commands af /check/

wellicht serverpath en server namen beslissen

input aflezen door de pins. input opslaan als een cordinaat, type. set settlement/ set city, x en y

knipper lichtjes  /check/

main:  
input aflezen en naar de server sturen
output aflezen van de server
leds werken


if info hex =/ hexdata = special error?



string naar array
*/