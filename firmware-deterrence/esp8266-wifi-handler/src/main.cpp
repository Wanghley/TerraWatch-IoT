#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// ==========================================
// === DEBUG CONFIGURATION ===
// ==========================================
// Set to true to see status messages on Computer
// Set to false if you suspect it confuses the Mega (though Mega should ignore them)
#define DEBUG_MODE false 

// Macro to handle printing only if debug is on
#if DEBUG_MODE
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

// ==========================================
// === WIFI / NETWORK SETTINGS ===
// ==========================================
const char *ssid = "liam";
const char *password = ""; // Set to "" or NULL for open networks

WiFiServer server(80);
WiFiUDP udp;

unsigned int broadcastPort = 4210;
unsigned int listenPort = 4211;
const int wifiIndicatorPin = 2; // Built-in LED (GPIO2)

bool broadcasting = true;
unsigned long lastBroadcast = 0;
const char* UNIQUE_ID = "GROUP2_DETER_ESP";
const char* STOP_MSG  = "STOP_BROADCAST";
const char* HEARTBEAT_MSG = "HEARTBEAT";

// Connection tracking
bool is_connected = false;
unsigned long lastHeartbeatTime = 0;
const unsigned long HEARTBEAT_INTERVAL = 5000;
const unsigned long HEARTBEAT_TIMEOUT = 10000;
const unsigned long HTTP_TIMEOUT = 5000;
String receiverIP = "";

void setup() {
  // IMPORTANT: 115200 must match the Mega's Serial1.begin
  Serial.begin(115200); 
  delay(100);

  DEBUG_PRINTLN("\n\n--- ESP8266 STARTING ---");
  
  pinMode(wifiIndicatorPin, OUTPUT);
  digitalWrite(wifiIndicatorPin, HIGH); // LED OFF (Active Low)

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  
  DEBUG_PRINT("[DEBUG] Connecting to WiFi: ");
  DEBUG_PRINTLN(ssid);
  
  if (password && strlen(password) > 0) {
    WiFi.begin(ssid, password);
  } else {
    WiFi.begin(ssid);
  }
  
  unsigned long wifiStartTime = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    DEBUG_PRINT(".");
    if (millis() - wifiStartTime > 10000) {
      DEBUG_PRINTLN("\n[DEBUG] WiFi Timeout! Restarting...");
      ESP.restart();
    }
  }

  DEBUG_PRINTLN("\n[DEBUG] WiFi Connected!");
  DEBUG_PRINT("[DEBUG] IP Address: ");
  DEBUG_PRINTLN(WiFi.localIP());

  digitalWrite(wifiIndicatorPin, LOW); // LED ON
  
  udp.begin(listenPort);
  DEBUG_PRINTLN("[DEBUG] UDP Listener Started");
  
  server.begin();
  DEBUG_PRINTLN("[DEBUG] HTTP Server Started");
}

void sendBroadcast() {
  DEBUG_PRINTLN("[DEBUG] Sending Broadcast...");
  udp.beginPacket("255.255.255.255", broadcastPort);
  String message = String(UNIQUE_ID) + "|" + WiFi.localIP().toString();
  udp.print(message);
  udp.endPacket();
}

void sendHeartbeat() {
  if (receiverIP.length() == 0) return;
  // DEBUG_PRINTLN("[DEBUG] Sending Heartbeat"); // Commented out to reduce spam
  udp.beginPacket(receiverIP.c_str(), broadcastPort);
  udp.print(HEARTBEAT_MSG);
  udp.endPacket();
}

void loop() {
  // --- WIFI RECONNECT LOGIC ---
  if (WiFi.status() != WL_CONNECTED) {
    DEBUG_PRINTLN("[DEBUG] WiFi Lost! Reconnecting...");
    is_connected = false;
    if (password && strlen(password) > 0) {
      WiFi.begin(ssid, password);
    } else {
      WiFi.begin(ssid);
    }
    delay(1000);
    return;
  }

  // --- UDP HANDSHAKE LOGIC (FINDING THE RECEIVER) ---
  if (!is_connected) {
    if (millis() - lastBroadcast > 1000) {
      sendBroadcast();
      lastBroadcast = millis();
    }
    
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';
      
      DEBUG_PRINT("[DEBUG] UDP Packet received: ");
      DEBUG_PRINTLN(buffer);

      if (strcmp(buffer, STOP_MSG) == 0) {
        receiverIP = udp.remoteIP().toString();
        is_connected = true;
        lastHeartbeatTime = millis();
        DEBUG_PRINTLN("[DEBUG] CONNECTION ESTABLISHED with: " + receiverIP);
      }
    }
    delay(10);
    return;
  }

  // --- HEARTBEAT LOGIC ---
  if (millis() - lastBroadcast >= HEARTBEAT_INTERVAL) {
    sendHeartbeat();
    lastBroadcast = millis();
  }
  
  // Read UDP ACKs
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char buffer[256];
    int len = udp.read(buffer, 255);
    if (len > 0) buffer[len] = '\0';
    String senderAddrStr = udp.remoteIP().toString();

    if (strcmp(buffer, "HEARTBEAT_ACK") == 0 && senderAddrStr == receiverIP) {
      lastHeartbeatTime = millis();
      // DEBUG_PRINTLN("[DEBUG] Heartbeat ACK received");
    } else if (strcmp(buffer, STOP_MSG) == 0) {
      receiverIP = senderAddrStr;
      lastHeartbeatTime = millis();
      DEBUG_PRINTLN("[DEBUG] STOP_MSG received (Refresh connection)");
    }
  }

  if (millis() - lastHeartbeatTime > HEARTBEAT_TIMEOUT) {
    DEBUG_PRINTLN("[DEBUG] Heartbeat Timeout! Resetting connection...");
    is_connected = false;
    return;
  }

  // --- HTTP SERVER LOGIC ---
  WiFiClient client = server.available();
  if (!client) return;

  DEBUG_PRINTLN("[DEBUG] New HTTP Client Connected");

  unsigned long clientConnectTime = millis();
  while (!client.available()) {
    delay(100);
    if (millis() - clientConnectTime > HTTP_TIMEOUT) { 
      DEBUG_PRINTLN("[DEBUG] HTTP Client Timeout");
      client.stop(); 
      return; 
    }
  }

  String req = client.readStringUntil('\r');
  client.read(); 
  
  DEBUG_PRINT("[DEBUG] Request: ");
  DEBUG_PRINTLN(req);

  // Parse Content-Length
  int contentLength = 0;
  while (client.available()) {
    String line = client.readStringUntil('\r');
    if (client.available()) client.read();
    if (line.length() <= 1) break; 
    if (line.startsWith("Content-Length:")) {
      contentLength = line.substring(15).toInt();
    }
  }

  String payload = "";
  if (req.startsWith("POST") && contentLength > 0) {
    unsigned long pStart = millis();
    while (client.available() < contentLength && (millis() - pStart < 2000)) delay(10);
    for (int i = 0; i < contentLength; i++) payload += (char)client.read();
    
    DEBUG_PRINT("[DEBUG] Payload: ");
    DEBUG_PRINTLN(payload);
  }

  // --- SEND COMMAND TO ARDUINO MEGA ---
  if (req.startsWith("POST")) {
    DEBUG_PRINTLN("[DEBUG] >> SENDING COMMAND TO MEGA >>");
    
    if (payload == "light-trigger") {
      // This is the actual command the Mega listens for:
      Serial.println("CMD_LIGHT"); 
      DEBUG_PRINTLN("[DEBUG] Sent: CMD_LIGHT");
    } else {
      // This is the actual command the Mega listens for:
      Serial.println("CMD_DETER"); 
      DEBUG_PRINTLN("[DEBUG] Sent: CMD_DETER");
    }
  }

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain");
  client.println("Connection: close");
  client.println();
  client.println("ACK");
  
  delay(10);
  client.stop();
  DEBUG_PRINTLN("[DEBUG] Client Disconnected");
  
  lastHeartbeatTime = millis();
}