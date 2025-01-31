#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// put function declarations here:
const char* ssid = "WIFISSID";
const char* password = "WIFIPASSWORD";

//Your Domain name with URL path or IP address with path
String serverName = "http://192.168.137.208:5000";

// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastTime = 0;
// Timer set to 10 minutes (600000)
//unsigned long timerDelay = 600000;
// Set timer to 5 seconds (5000)
unsigned long timerDelay = 5000;

void setup() {
  Serial.begin(9600); 

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
 
  Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");
}

void loop() {
  //Send an HTTP POST request every 10 minutes
  if ((millis() - lastTime) > timerDelay) {
    //Check WiFi connection status
    if(WiFi.status()== WL_CONNECTED){
      HTTPClient http;

      String serverPath = serverName + "?temperature=24.37";
      
      // Your Domain name with URL path or IP address with path
      http.begin(serverPath.c_str());
      
      // If you need Node-RED/server authentication, insert user and password below
      //http.setAuthorization("REPLACE_WITH_SERVER_USERNAME", "REPLACE_WITH_SERVER_PASSWORD");
      
      // Send HTTP GET request
      int serverchange = "iwha 123";
      http.PUT(serverchange);
      // Free resources
      http.end();
    }
    else {
      Serial.println("WiFi Disconnected");
    }
    lastTime = millis();
  }
}


//https put command
//https://www.w3schools.com/tags/ref_httpmethods.asp#:~:text=The%20POST%20Method,Host%3A%20w3schools.com <-
//https://github.com/microsoft/windows-universal-samples/tree/main/Samples/HttpClient