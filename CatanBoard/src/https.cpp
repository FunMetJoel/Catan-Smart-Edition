#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// put function declarations here:
const char* ssid = "WIFISSID";
const char* password = "WIFIPASSWORD";

//Your Domain name with URL path or IP address with path
IpAdress() {
  String serverRoad = "http://192.168.137.230:5000";
  String serverHuis = "" ;
  String serverHex =  "" ;
}
unsigned long lastTime = 0;

unsigned long timerDelay = 5000;

void starthttps() {
  Serial.begin(9600); 

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

}

void ServerPost()
http.POST("test")



//void ServerGet()
void ServerGet() {
  //Send an HTTP POST request every 10 minutes
  if ((millis() - lastTime) > timerDelay) {
    //Check WiFi connection status
    if(WiFi.status()== WL_CONNECTED){
      HTTPClient http;

      String serverPath = IpAdress(Server) + "?temperature=24.37";
      //server = 1,2 or 3...
      // Your Domain name with URL path or IP address with path
      http.begin(serverPath.c_str());
      
      // If you need Node-RED/server authentication, insert user and password below
      //http.setAuthorization("REPLACE_WITH_SERVER_USERNAME", "REPLACE_WITH_SERVER_PASSWORD");
      
      // Send HTTP GET request
      int httpResponseCode = http.GET();            /
      
      if (httpResponseCode>0) {
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        String payload = http.getString();              /*<--*/
        Serial.println(payload);
      }
      else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);
      }
      // Free resources
      http.end();
    }
    else {
      Serial.println("WiFi Disconnected");
    }
    lastTime = millis();
  }
}

/*to do: getstring verdelen in string straat, string dorp, string stad variabeles
http give requests op een webserver?

main aanpassen:
loop:
- http request vragen, http naar leds, leds aan, pins detecten, http put command sturen

pins detecten:
x en y = location
if power(x) = true & power(y) = true 
locationplayer = true


*/
//notes en namen aanpassen <- laatste. 
