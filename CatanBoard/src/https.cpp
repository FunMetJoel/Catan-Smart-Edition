#include "https.h"
#include <Arduino.h>
#include "catanState.h"
#include <WiFi.h>
#include <HTTPClient.h>

bool error = false;
// put function declarations here:
const char* ssid = "Klinki";
const char* password = "KlinkiSnek";

String serverRoad = "http://192.168.1.35:5000";
String serverHouse = "http://192.168.1.35:5000"; //even van jowel vragen
String serverHex = "http://192.168.1.35:5000";
String serverCity = "http://192.168.1.35:5000";


void setupHttp() {

  WiFi.begin(ssid, password);
  if(WiFi.status() != WL_CONNECTED) {
    bool wificonnect = true; //<-deze kan weg

  }   
  else {
    error = true;
  }
}

void getRoad() {
  HTTPClient http;
  String serverPath = serverRoad + "?temperature=24.37";
  http.begin(serverPath.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String Roaddata = http.getString();
  }
//  else {
//error message
//  }
}

void getHouse() {
  HTTPClient http;
  String serverPath = serverHouse + "?temperature=24.37";
  http.begin(serverPath.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String Housedata = http.getString();
    for (int i = 0; i < 54; i++){
      &statePointer->corners[i].player = Housedata[i * 2];
      &statePointer->corners[i].level = Housedata[i * 2 + 1];
    }
  }
  else {
    error = true
  }
}

void getHex() {
  HTTPClient http;
  Serial.println(serverHex.c_str());
  http.begin(serverHex.c_str());
  // String serverPath = serverHex + "?temperature=24.37";
  // http.begin(serverPath.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String Hexdata = http.getString();
  }
//  else {
//error message
//  }
}

void putRoad() {
//get httppath = url/welke road bouwen serverHex + "/71"
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverHexPutPath = serverHex.c_str() +  //number; //<- 
    http.begin(serverHexPutPath.c_str());
    int httpResponseCode = http.GET();
  }
}

void putHouse() {
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverHexPutPath = serverHouse.c_str() +  //number; //<- 
    http.begin(serverHexPutPath.c_str());
    int httpResponseCode = http.GET();
  }
  //response code get?
}

void putCity() {
  int httpResponseCode = 0;
  while (httpResponseCode <= 0) {
    HTTPClient http;
    String serverCityPutPath = serverCity.c_str() +  //number; //<- 
    http.begin(serverCityPutPath.c_str());
    int httpResponseCode = http.GET();
  }
}

void getCurrentPlayer() {
  HTTPClient http;
  http.begin(serverHex.c_str());
  int httpResponseCode = http.GET();
  if (httpResponseCode>0) {
    String currentPlayer = http.getString(); 
  }
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



main: while (error = false) {

}
else {
  error()
  restart
}

string naar array
*/