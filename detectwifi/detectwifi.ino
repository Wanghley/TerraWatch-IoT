#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "ECE449deco";
const char* password = "ece449$$";

WiFiUDP udp;

const int esp32pin = 32;
const int orangepipin = 33;
bool esp32LastPinState = LOW;
bool orangepiLastPinState = LOW; 

const unsigned int listenPort = 4210;       // Listening for broadcast
const unsigned int senderResponsePort = 4211;  // Port where sender listens for STOP

const char* UNIQUE_ID = "GROUP2_DETER_ESP";
const char* STOP_MSG  = "STOP_BROADCAST";

String senderIP;

void setup() {
  Serial.begin(115200);
  pinMode(esp32pin, INPUT_PULLDOWN);
  pinMode(orangepipin, INPUT_PULLDOWN);
  
  // -----------------------------
  // Connect to WiFi
  // -----------------------------
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }

  Serial.println("\nConnected!");
  Serial.print("Receiver IP: ");
  Serial.println(WiFi.localIP());

  // -----------------------------
  // Start UDP listening
  // -----------------------------
  udp.begin(listenPort);
  Serial.println("UDP Receiver Ready. Listening...");

  // -----------------------------
  // Main logic loop – stays in setup()
  // -----------------------------
  while (true) {

    int packetSize = udp.parsePacket();
    if (packetSize) {
      // Read the incoming UDP packet
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';

      String msg = String(buffer);
      Serial.print("Received broadcast: ");
      Serial.println(msg);

      // Expecting: UNIQUE_ID|<IP>
      int sepIndex = msg.indexOf('|');
      if (sepIndex == -1) continue;

      String id = msg.substring(0, sepIndex);
      senderIP = msg.substring(sepIndex + 1);

      Serial.print("Parsed Sender IP: ");
      Serial.println(senderIP);

      // If ID matches, send STOP response
      if (id == UNIQUE_ID) {

        for(int i = 0; i < 5; i++){
        Serial.println("Identifier matched. Sending STOP...");

        udp.beginPacket(senderIP.c_str(), senderResponsePort);
        udp.print(STOP_MSG);
        udp.endPacket();

        Serial.println("STOP message sent.");
        delay(500);
        }
        break;
      }
    }

    delay(10); // prevent watchdog reset
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  // COMMENT CONDITION AND UNCOMMENT DELAY
  if((digitalRead(esp32pin) == HIGH && esp32LastPinState == LOW) || (digitalRead(orangepipin) && orangepiLastPinState == LOW)){
    Serial.println("Pin went HIGH -> Sending POST");

    WiFiClient client;
    if (!client.connect(senderIP.c_str(), 80)) {
      Serial.println("Connection failed");
      delay(300);
      return;
    }

    String payload = "trigger=1";

    // --- POST REQUEST ---
    client.println("POST / HTTP/1.1");
    client.println("Host: " + senderIP);
    client.println("Content-Type: text/plain");
    client.println("Content-Length: " + String(payload.length()));
    client.println();
    client.print(payload);

    // --- READ RESPONSE ---
    unsigned long timeout = millis();
    while (!client.available()) {
      if (millis() - timeout > 3000) {
        Serial.println("Timeout waiting for response");
        client.stop();
        return;
      }
    }

    while (client.available()) {
      String line = client.readStringUntil('\n');
      Serial.println("Response: " + line);
    }

    client.stop();
    Serial.println("POST complete");

    // String url = "http://" + senderIP + "/";

    // HTTPClient http;
    // http.begin(url);

    // String payload = "{\"pin\":\"HIGH\"}";
    // http.addHeader("Content-Type", "application/json");

    // int code = http.POST(payload);

    // Serial.print("POST response code: ");
    // Serial.println(code);

    // if (code > 0) {
    //   Serial.println("Response: " + http.getString());
    // }

    // http.end();
   }
   esp32LastPinState = digitalRead(esp32pin);
   orangepiLastPinState = digitalRead(orangepipin);

}
