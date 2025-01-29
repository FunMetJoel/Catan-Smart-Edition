#include "https.h"
#include <Arduino.h>
#include "catanState.h"
#include <WiFi.h>
#include <HTTPClient.h>


// put function declarations here:
const char* ssid = "WIFISSID";
const char* password = "WIFIPASSWORD";

String serverRoad = "http://192.168.137.208:5000";
String serverHouse = "http://192.168.137.208:5000"; //even van jowel vragen
String serverHex = "http://192.168.137.208:5000";


void setupHttp( ) {
Serial.begin(9600); 

  WiFi.begin(ssid, password);
  if(WiFi.status() != WL_CONNECTED) {
    bool wificonnect = true;

  }
//    licht effecten?? knipper lichies    
//  else {
//error message
//  }
}

//in main, if(WiFi.status()== WL_CONNECTED) dan connect get
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
  }
//  else {
//error message
//  }
}

void getHex() {
  HTTPClient http;
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
}

void putHouse() {
  
}

void putHex() {
  
}

// void currentplayer() {
//
//}



/*ToDo--------------
- put commands af

wellicht serverpath en server namen beslissen

input aflezen door de pins. input opslaan als een cordinaat, type. set settlement/ set city, x en y

main:  
input aflezen en naar de server sturen
output aflezen van de server
leds werken
*/