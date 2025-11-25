
#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_VS1053.h>
#include <Arduino.h>
#include <ArduinoJson.h>
#include "esp_system.h" //random sound change

// L298N pin assignments
#define ENA 5
#define IN1 4
#define IN2 7
#define ENB 9
#define IN3 10
#define IN4 14

// === PIN CONFIGURATION (confirmed working) ===
#define VS1053_RST  -1   // not wired
#define VS1053_CS    47   // VS1053 XCS


#define VS1053_DCS   48   // VS1053 XDCS
#define VS1053_DREQ  21   // VS1053 DREQ
#define SD_CS       20   // SD card CS

// SPI hardware pins (ESP32 VSPI)
#define SPI_SCK   18
#define SPI_MISO  19
#define SPI_MOSI  17

// === File to play (use absolute path) ===
const char *tracks[] = { //random sound change
"/bear.mp3",
"/monster.mp3",
"/eagle.mp3",
"/owl.mp3",
"/dog.mp3",
};

const size_t TRACK_COUNT = 5;

// === Create player object ===
Adafruit_VS1053_FilePlayer player(VS1053_RST, VS1053_CS, VS1053_DCS, VS1053_DREQ, SD_CS);

// ----- USER SETTINGS -----
char direction = 'F';  // 'F' for forward, 'R' for reverse
int speedValue = 255;  // 0–255 (higher = faster)
int runSeconds = 5;    // how long to run, in seconds
// --------------------------

// Full-step sequence (4 steps per cycle)
int stepsSeq[4][4] = {
  {1, 0, 1, 0},
  {0, 1, 1, 0},
  {0, 1, 0, 1},
  {1, 0, 0, 1}
};

int stepDelayMs = 5;

const int relayPin = 16;  // pin controlling relay
const bool RELAY_ACTIVE_LOW = true;  // change to false if your relay is active HIGH

const char *ssid = "ECE449deco";
const char *password = "ece449$$";

WiFiServer server(80);
WiFiUDP udp;

unsigned int broadcastPort = 4210;  // Port to send broadcasts to
unsigned int listenPort = 4211;     // Port to listen for STOP messages

bool broadcasting = true;
unsigned long lastBroadcast = 0;

const char* UNIQUE_ID = "GROUP2_DETER_ESP";
const char* STOP_MSG  = "STOP_BROADCAST";
const char* HEARTBEAT_MSG = "HEARTBEAT";

// Connection tracking
bool is_connected = false;
unsigned long lastHeartbeatTime = 0;
const unsigned long HEARTBEAT_INTERVAL = 5000;  // Send heartbeat every 5 seconds
const unsigned long HEARTBEAT_TIMEOUT = 10000;  // 10 second timeout
const unsigned long HTTP_TIMEOUT = 5000;        // 5 second HTTP timeout
String receiverIP = "";  // IP of the detectwifi board

// Action triggers updated by POST requests
bool newActionReceived = false;
String lastPostPayload = "";

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Starting MP3 loop player...");

  // Initialize SPI & SD
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  if (!SD.begin(SD_CS, SPI)) {
    Serial.println("SD card failed!");
    while (1);
  }

  // Initialize VS1053
  if (!player.begin()) {
    Serial.println("VS1053 not found!");
    while (1);
  }

  // Relay for light
  pinMode(relayPin, OUTPUT);
  
  Serial.println();
  Serial.print("[WiFi] Connecting to ");
  Serial.println(ssid);

  // Auto reconnect is set true as default
  // To set auto connect off, use the following function
  //    WiFi.setAutoReconnect(false);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }

  Serial.println("\nConnected!");
  Serial.print("My IP: ");
  Serial.println(WiFi.localIP());

  // Begin UDP listening for STOP message and heartbeats
  udp.begin(listenPort);
  Serial.print("UDP listening on port ");
  Serial.println(listenPort);

  // Broadcast until we receive STOP message
  while(true){
    // Check WiFi connection status
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected! Setting is_connected = false");
      is_connected = false;
      // Attempt to reconnect
      WiFi.begin(ssid, password);
      delay(1000);
      continue;
    }

    // While not connected, broadcast every 1 second
    if (!is_connected && (millis() - lastBroadcast > 1000)) {
      sendBroadcast();
      lastBroadcast = millis();
    }

    // Check if any message is received
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';

      // Get sender IP
      IPAddress senderIP = udp.remoteIP();
      
      Serial.print("Received from ");
      Serial.print(senderIP);
      Serial.print(": ");
      Serial.println(buffer);

      if (strcmp(buffer, STOP_MSG) == 0) {
        Serial.println("STOP message received! Connection established.");
        receiverIP = senderIP.toString();
        is_connected = true;
        lastHeartbeatTime = millis();
        break;
      }
    }
    
    delay(100); // Small delay to prevent watchdog reset
  }
  
  Serial.println("server begins");
  
  server.begin();

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);


  // Set moderate volume (0 loudest)
  // Use this snippet in your existing working sketch (after player.begin())
  player.setVolume(0, 0);        // desired volume
  // Enable interrupt-driven playback. On many platforms the library defines the constant:
  #if defined(VS1053_FILEPLAYER_PIN_INT)
    player.useInterrupt(VS1053_FILEPLAYER_PIN_INT);
  #else
    // fallback: use the DREQ pin number directly
    player.useInterrupt(VS1053_DREQ); 
  #endif

  // Optional: use interrupt-driven playback (ESP32 can handle without)
  // player.useInterrupt(VS1053_FILEPLAYER_PIN_INT);

  randomSeed(analogRead(3));  // randomize using floating analog input

  Serial.println("Initialization complete.");
}

void loop() {
  // Check WiFi connection status
  if (WiFi.status() != WL_CONNECTED) {
    if (is_connected) {
      Serial.println("WiFi disconnected! Setting is_connected = false");
      is_connected = false;
      receiverIP = "";
    }
    // Attempt to reconnect
    WiFi.begin(ssid, password);
    delay(1000);
    return;
  }

  // Handle reconnection logic when not connected
  if (!is_connected) {
    // Broadcast every 1 second until we get STOP message
    if (millis() - lastBroadcast > 1000) {
      sendBroadcast();
      lastBroadcast = millis();
    }

    // Check for STOP message
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';

      IPAddress senderIP = udp.remoteIP();
      
      if (strcmp(buffer, STOP_MSG) == 0) {
        Serial.println("STOP message received! Connection re-established.");
        receiverIP = senderIP.toString();
        is_connected = true;
        lastHeartbeatTime = millis();
      }
    }
    delay(10);
    return;
  }

  // When connected, handle heartbeats and HTTP server
  
  // Send heartbeat periodically
  if (millis() - lastBroadcast >= HEARTBEAT_INTERVAL) {
    sendHeartbeat();
    lastBroadcast = millis();
  }

  // Check for heartbeat responses
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char buffer[256];
    int len = udp.read(buffer, 255);
    if (len > 0) buffer[len] = '\0';

    IPAddress senderAddr = udp.remoteIP();
    String senderAddrStr = senderAddr.toString();

    if (strcmp(buffer, "HEARTBEAT_ACK") == 0 && senderAddrStr == receiverIP) {
      lastHeartbeatTime = millis();
      Serial.println("Heartbeat ACK received from " + receiverIP);
    } else if (strcmp(buffer, STOP_MSG) == 0) {
      // Receiver wants to reconnect
      receiverIP = senderAddrStr;
      lastHeartbeatTime = millis();
      Serial.println("STOP message received - connection refreshed from " + receiverIP);
    }
  }

  // Check heartbeat timeout
  if (millis() - lastHeartbeatTime > HEARTBEAT_TIMEOUT) {
    Serial.println("Heartbeat timeout! Setting is_connected = false");
    is_connected = false;
    receiverIP = "";
    return;
  }

  // Handle HTTP server
  WiFiClient client = server.available();
  if (!client) {
    delay(10);
    return;
  }

  Serial.println("Client connected");
  unsigned long clientConnectTime = millis();

  // Wait for header with timeout
  while (!client.available()) {
    delay(100);
    // Check for HTTP timeout
    if (millis() - clientConnectTime > HTTP_TIMEOUT) {
      Serial.println("HTTP timeout waiting for request!");
      is_connected = false;
      receiverIP = "";
      client.stop();
      return;
    }
    // Also check connection status
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi disconnected during HTTP request!");
      is_connected = false;
      receiverIP = "";
      client.stop();
      return;
    }
  }

  String req = client.readStringUntil('\r');
  Serial.println("Request:");
  Serial.println(req);

  // Read full header until blank line and extract Content-Length
  int contentLength = 0;
  while (client.available()) {
    String line = client.readStringUntil('\r');
    if (line.startsWith("Content-Length:")) {
      // Extract content length value
      int colonIndex = line.indexOf(':');
      if (colonIndex != -1) {
        String lengthStr = line.substring(colonIndex + 1);
        lengthStr.trim();
        contentLength = lengthStr.toInt();
      }
    }
    if (line == "\n" || line == "\r\n") break;
  }

  // Read POST body if present
  String payload = "";
  if (req.startsWith("POST") && contentLength > 0) {
    // Wait for payload to arrive
    unsigned long payloadStartTime = millis();
    while (client.available() < contentLength && (millis() - payloadStartTime < HTTP_TIMEOUT)) {
      delay(10);
    }
    
    // Read the exact number of bytes specified by Content-Length
    if (client.available() >= contentLength) {
      payload = "";
      for (int i = 0; i < contentLength && client.available(); i++) {
        char c = client.read();
        payload += c;
      }
      payload.trim(); // Remove any trailing whitespace
      Serial.print("POST payload received: ");
      Serial.println(payload);
    }
  }

  // -----------------------------------
  //   PROCESS POST REQUEST
  // -----------------------------------
  if (req.startsWith("POST")) {
    Serial.println("POST request detected");
    Serial.print("Payload: ");
    Serial.println(payload.length() > 0 ? payload : "(empty)");
    
    if (payload == "light-trigger") {
      Serial.println("Light trigger detected - keeping lights on for 8 seconds");
      // Use lightFlicker to keep lights on for 8 seconds (8000ms on, 0ms off)
      lightFlicker(8000, 0);
      Serial.println("Lights turned off");
    } else if (payload == "trigger=1" || payload.length() == 0) {
      // Default deterrent behavior for trigger=1 or empty payload
      Serial.println("Deterrent trigger detected - running full deterrent sequence");
      deterrent();
    } else {
      // Unknown payload - still run deterrent as default
      Serial.print("Unknown payload '");
      Serial.print(payload);
      Serial.println("' - running default deterrent");
      deterrent();
    }
  }

  // -----------------------------------
  //   SEND HTTP RESPONSE
  // -----------------------------------
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain");
  client.println("Connection: close");
  client.println();
  client.println("ACK from Sender ESP32");

  // Wait for response to be sent with timeout
  unsigned long responseStartTime = millis();
  while (client.connected() && (millis() - responseStartTime < HTTP_TIMEOUT)) {
    delay(100);
  }

  delay(10);
  client.stop();
  Serial.println("Client disconnected");
  
  // Reset heartbeat timer after successful HTTP interaction
  lastHeartbeatTime = millis();
}

void deterrent(){
  if (!player.playingMusic) {
        const char* MP3_FILE = tracks[pickRandomTrack()];
        Serial.println("Connected to client Starting playback");
        player.startPlayingFile(MP3_FILE);
      }


      for (int i=0; i<5; i++) {
        // Random ON duration between 200ms and 2000ms
        int onTime = random(200, 2000);
        // Random OFF duration between 200ms and 3000ms
        int offTime = random(200, 3000);
        lightFlicker(onTime, offTime);
      }

      
      //START MOTOR
      analogWrite(ENA, 255);
      analogWrite(ENB, 255);

      motorCall();

      if(player.playingMusic) {
        player.stopPlaying();
      }
      Serial.println("Exiting deter");
}

void stepForward() {
  for (int i = 0; i < 4; i++) {
    setStep(stepsSeq[i]);
    delay(stepDelayMs);
  }
}

void stepBackward() {
  for (int i = 3; i >= 0; i--) {
    setStep(stepsSeq[i]);
    delay(stepDelayMs);
  }
}

void setStep(int pins[4]) {
  digitalWrite(IN1, pins[0]);
  digitalWrite(IN2, pins[1]);
  digitalWrite(IN3, pins[2]);
  digitalWrite(IN4, pins[3]);
}

void stopStepper() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
// helper function to send a broadcast message
void sendBroadcast() {
  udp.beginPacket("255.255.255.255", broadcastPort);

  // Message format:
  //     UNIQUE_ID|IP_ADDRESS
  String message = String(UNIQUE_ID) + "|" + WiFi.localIP().toString();

  udp.print(message);
  udp.endPacket();

  Serial.println("Broadcast message sent: " + message);
}

// helper function to send a heartbeat message
void sendHeartbeat() {
  if (receiverIP.length() == 0) return;
  
  udp.beginPacket(receiverIP.c_str(), broadcastPort);  // Send to port 4210 where detectwifi listens
  udp.print(HEARTBEAT_MSG);
  udp.endPacket();
  
  Serial.println("Heartbeat sent to " + receiverIP);
}

int pickRandomTrack() {
  uint32_t seed = esp_random();
  randomSeed(seed);
  return random(0, TRACK_COUNT);
}

void lightFlicker(int onDuration, int offDuration){
  digitalWrite(relayPin, HIGH);
  delay(onDuration);
  digitalWrite(relayPin, LOW);
  delay(offDuration);
}

void runDirectionFor(unsigned long durationMs, bool forward, int speed) {
  if (speed > 180) speed = 180;            // safety cap
  if (speed < 0) speed = 0;
  speedValue = speed;
  stepDelayMs = map(speedValue, 0, 255, 10, 1);

  unsigned long startTime = millis();
  while (millis() - startTime < durationMs) {
    if (forward) {
      stepForward();
    } else {
      stepBackward();
    }
  }
}

void motorCall() {
  int choice = random(5); // 0..4
  Serial.print("Preset chosen: ");
  Serial.println(choice);

  switch (choice) {
    case 0:
      // Preset 0: steady forward 4s at max-safe speed
      Serial.println("Preset 0: forward 4s @180");
      runDirectionFor(4000UL, true, 180);
      delay(333);
      break;

    case 1:
      // Preset 1: 2s forward fast, 2s backward medium
      Serial.println("Preset 1: forward 2s @160, backward 2s @120");
      runDirectionFor(2000UL, true, 160);
      delay(333);
      runDirectionFor(2000UL, false, 120);
      delay(333);
      break;

    case 2:
      // Preset 2: four 1s bursts alternating forward/back at two speeds
      Serial.println("Preset 2: fwd 1s@140, back 1s@140, fwd 1s@90, back 1s@90");
      runDirectionFor(1000UL, true, 140);
      delay(333);
      runDirectionFor(1000UL, false, 140);  
      delay(333);
      runDirectionFor(1000UL, true, 90);
      delay(333);
      runDirectionFor(1000UL, false, 90);
      break;

    case 3:
      // Preset 3: ramp/hold style: gentle -> fast forward, then fast -> gentle backward
      Serial.println("Preset 3: fwd 1s@80, fwd 1s@140, back 1s@140, back 1s@80");
      runDirectionFor(1000UL, true, 80);
      delay(333);
      runDirectionFor(1000UL, true, 140);
      delay(333);
      runDirectionFor(1000UL, false, 140);  
      delay(333);
      runDirectionFor(1000UL, false, 80);
      delay(333);
      break;

    case 4:
      // Preset 4: short pulses: 4 alternating 0.5s pulses (total 4s = 8 pulses of 0.5s? we'll do 8*0.5)
      Serial.println("Preset 4: 8 x 0.5s pulses alternating, speeds 180/120");
      // We'll do 8 pulses of 500ms (alternating direction and alternating speeds)
      for (int i = 0; i < 8; ++i) {
        bool forward = (i % 2 == 0);
        int speed = (i % 2 == 0) ? 180 : 120;
        runDirectionFor(500UL, forward, speed);
        delay(333);
      }
      break;
  }

  Serial.println("Preset complete.");
}
