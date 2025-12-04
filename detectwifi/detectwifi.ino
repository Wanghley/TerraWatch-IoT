#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "ECE449deco";
const char* password = "ece449$$";

WiFiUDP udp;

const int lightPin = 14;
const int httpRequestIndicatorPin = 26; // red LED  
const int wifiIndicatorPin = 23; // blue LED
const int orangepipin = 33;
bool lightLastPinState = LOW;
bool orangepiLastPinState = LOW; 
bool httpIndicatorState = LOW;

const unsigned int listenPort = 4210;       // Listening for broadcast
const unsigned int senderResponsePort = 4211;  // Port where sender listens for STOP

const char* UNIQUE_ID = "GROUP2_DETER_ESP";
const char* STOP_MSG  = "STOP_BROADCAST";
const char* HEARTBEAT_MSG = "HEARTBEAT";
const char* HEARTBEAT_ACK = "HEARTBEAT_ACK";

// Connection tracking
bool is_connected = false;
unsigned long lastHeartbeatTime = 0;
const unsigned long HEARTBEAT_TIMEOUT = 10000;  // 10 second timeout
const unsigned long HTTP_TIMEOUT = 5.500;        // 5.5 second HTTP timeout

String senderIP;

void setup() {
  Serial.begin(115200);
  pinMode(lightPin, INPUT_PULLDOWN);
  pinMode(orangepipin, INPUT_PULLDOWN);
  pinMode(httpRequestIndicatorPin, OUTPUT);
  digitalWrite(httpRequestIndicatorPin, LOW);
  
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
  // Connection establishment loop – stays in setup()
  // -----------------------------
  while (!is_connected) {
    // Check WiFi connection status
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected! Setting is_connected = false");
      is_connected = false;
      senderIP = "";
      // Attempt to reconnect
      WiFi.begin(ssid, password);
      delay(1000);
      continue;
    }

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
        is_connected = true;
        lastHeartbeatTime = millis();
        Serial.println("Connection established!");
        break;
      }
    }

    delay(100); // prevent watchdog reset
  }
}

void toggleHttpIndicator() {
  httpIndicatorState = !httpIndicatorState;
  digitalWrite(httpRequestIndicatorPin, httpIndicatorState);
}

void sendPostRequest(String payload) {
  WiFiClient client;
  unsigned long connectStartTime = millis();
  
  // Attempt connection with timeout
  while (!client.connect(senderIP.c_str(), 80)) {
    if (millis() - connectStartTime > HTTP_TIMEOUT) {
      Serial.println("HTTP connection timeout!");
      is_connected = false;
      senderIP = "";
      if (httpIndicatorState == HIGH) {
        toggleHttpIndicator();
      }
      return;
    }
    delay(100);
    // Check WiFi status
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected during connection attempt!");
      is_connected = false;
      senderIP = "";
      if (httpIndicatorState == HIGH) {
        toggleHttpIndicator();
      }
      return;
    }
  }

  unsigned long requestStartTime = millis();

  // Toggle indicator for outgoing HTTP request
  toggleHttpIndicator();

  // --- POST REQUEST ---
  client.println("POST / HTTP/1.1");
  client.println("Host: " + senderIP);
  client.println("Content-Type: text/plain");
  client.println("Content-Length: " + String(payload.length()));
  client.println();
  client.print(payload);

  // --- READ RESPONSE WITH TIMEOUT ---
  unsigned long timeout = millis();
  while (!client.available()) {
    if (millis() - timeout > HTTP_TIMEOUT) {
      Serial.println("HTTP timeout waiting for response!");
      client.stop();
      is_connected = false;
      senderIP = "";
      // Toggle back to indicate end of HTTP activity on timeout
      toggleHttpIndicator();
      return;
    }
    delay(100);
    // Check WiFi status
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected during HTTP request!");
      is_connected = false;
      senderIP = "";
      client.stop();
      // Toggle back to indicate end of HTTP activity on WiFi loss
      toggleHttpIndicator();
      return;
    }
  }

  // Read response
  while (client.available()) {
    String line = client.readStringUntil('\n');
    Serial.println("Response: " + line);
  }

  client.stop();
  Serial.println("POST complete");
  
  // Toggle indicator for completed HTTP response
  toggleHttpIndicator();
  
  // Reset heartbeat timer after successful HTTP interaction
  lastHeartbeatTime = millis();
}

void loop() {
  // Check WiFi connection status
  if (WiFi.status() != WL_CONNECTED) {
    if (is_connected) {
      Serial.println("WiFi disconnected! Setting is_connected = false");
      is_connected = false;
      senderIP = "";
    }
    // Attempt to reconnect
    WiFi.begin(ssid, password);
    delay(1000);
    return;
  }

  // Handle reconnection logic when not connected
  if (!is_connected) {
    // Listen for broadcasts
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';

      String msg = String(buffer);
      Serial.print("Received broadcast: ");
      Serial.println(msg);

      // Expecting: UNIQUE_ID|<IP>
      int sepIndex = msg.indexOf('|');
      if (sepIndex != -1) {
        String id = msg.substring(0, sepIndex);
        senderIP = msg.substring(sepIndex + 1);

        if (id == UNIQUE_ID) {
          Serial.println("Identifier matched. Sending STOP...");
          for(int i = 0; i < 5; i++){
            udp.beginPacket(senderIP.c_str(), senderResponsePort);
            udp.print(STOP_MSG);
            udp.endPacket();
            delay(500);
          }
          is_connected = true;
          lastHeartbeatTime = millis();
          Serial.println("Connection re-established!");
        }
      }
    }
    delay(100);
    return;
  }

  // When connected, handle heartbeats and pin monitoring
  
  // Check for heartbeat messages from DeterESP32
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char buffer[256];
    int len = udp.read(buffer, 255);
    if (len > 0) buffer[len] = '\0';

    if (strcmp(buffer, HEARTBEAT_MSG) == 0) {
      // Respond to heartbeat - verify it's from our connected sender
      IPAddress senderAddr = udp.remoteIP();
      if (senderAddr.toString() == senderIP) {
        udp.beginPacket(senderIP.c_str(), senderResponsePort);  // Send ACK to port 4211 where DeterESP32 listens
        udp.print(HEARTBEAT_ACK);
        udp.endPacket();
        
        lastHeartbeatTime = millis();
        Serial.println("Heartbeat received and ACK sent");
      }
    }
  }

  // Check heartbeat timeout
  if (millis() - lastHeartbeatTime > HEARTBEAT_TIMEOUT) {
    Serial.println("Heartbeat timeout! Setting is_connected = false");
    is_connected = false;
    senderIP = "";
    return;
  }

  // Monitor pins and send POST when triggered
  bool lightCurrentState = digitalRead(lightPin);
  bool orangepiCurrentState = digitalRead(orangepipin);
  
  bool lightPinRising = (lightCurrentState == HIGH && lightLastPinState == LOW);
  bool orangepiPinRising = (orangepiCurrentState == HIGH && orangepiLastPinState == LOW);
  
  // Send light-trigger when light pin goes HIGH
  if (lightPinRising) {
    Serial.println("Light pin went HIGH -> Sending light-trigger POST");
    sendPostRequest("light-trigger");
  }
  // Send trigger=1 when orangepi pin goes HIGH (from LOW)
  else if (orangepiPinRising) {
    Serial.println("OrangePi pin went HIGH -> Sending trigger=1 POST");
    sendPostRequest("trigger=1");
  }
  
  // Update pin states for next iteration
  lightLastPinState = lightCurrentState;
  orangepiLastPinState = orangepiCurrentState;
  
  delay(10);
}
