#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_VS1053.h>
#include <Arduino.h>
#include <ArduinoJson.h>

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
const char *MP3_FILE = "/track001.mp3";

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

// IPAddress local_IP(192, 168, 68, 10);   
// IPAddress gateway(192, 168, 68, 1);
// IPAddress subnet(255, 255, 255, 0);

WiFiServer server(80);
WiFiUDP udp;

unsigned int udpPort = 4210; 
bool broadcasting = true;
unsigned long lastBroadcast = 0;
String deviceName = "GROUP2_DETER_ESP";

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

  WiFi.begin(ssid, password);
  // Auto reconnect is set true as default
  // To set auto connect off, use the following function
  //    WiFi.setAutoReconnect(false);

  // Will try for about  seconds (30x 800ms)
  int tryDelay = 800;
  int numberOfTries = 30;

  // Wait for the WiFi event
  while (true) {

    switch (WiFi.status()) {
      case WL_NO_SSID_AVAIL: Serial.println("[WiFi] SSID not found"); break;
      case WL_CONNECT_FAILED:
        Serial.print("[WiFi] Failed - WiFi not connected! Reason: ");
        return;
        break;
      case WL_CONNECTION_LOST: Serial.println("[WiFi] Connection was lost"); break;
      case WL_SCAN_COMPLETED:  Serial.println("[WiFi] Scan is completed"); break;
      case WL_DISCONNECTED:    Serial.println("[WiFi] WiFi is disconnected"); break;
      case WL_CONNECTED:
        Serial.println("[WiFi] WiFi is connected!");
        Serial.print("[WiFi] IP address: ");
        Serial.println(WiFi.localIP());
        break;
      default:
        Serial.print("[WiFi] WiFi Status: ");
        Serial.println(WiFi.status());
        break;
    }
    if (WiFi.status() == WL_CONNECTED) {
      break;
    }
    delay(tryDelay);

    if (numberOfTries <= 0) {
      Serial.print("[WiFi] Failed to connect to WiFi! Restarting.");
      // Use disconnect function to force stop trying to connect
      WiFi.disconnect();
      ESP.restart();
      return;
    } else {
      numberOfTries--;
    }
  }
  udp.begin(udpPort);
  Serial.println("UDP ready.");

  while(true){
    // Broadcast every 5 seconds
    if (broadcasting && millis() - lastBroadcast > 5000) {
      sendBroadcast();
      lastBroadcast = millis();
    }

    // Check if any message is received
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char incoming[255];
      int len = udp.read(incoming, 254);
      if (len > 0) incoming[len] = '\0';

      Serial.printf("Received message: %s\n", incoming);

    // Parse incoming JSON
    StaticJsonDocument<128> doc;
    DeserializationError error = deserializeJson(doc, incoming);
    if (!error && doc["type"] == "stop") {
      broadcasting = false;
      Serial.println("Received STOP command. Halting broadcast.");
      break;
      }
    }
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

  Serial.println("Initialization complete. Starting playback...");
}

void loop() {
  // When finished, restart

  
  WiFiClient client = server.available();
  if (client) {
    Serial.println("inside if");
    // close the connection:
    client.stop();

    if (!player.playingMusic) {
      Serial.println("Connected to client Starting playback");
      player.startPlayingFile(MP3_FILE);
      Serial.println("problem is with mp3 file");
    }

    // Random ON duration between 200ms and 2000ms
    int onTime = random(200, 2000);

    // Random OFF duration between 200ms and 3000ms
    int offTime = random(200, 3000);

    digitalWrite(relayPin, LOW);
    delay(onTime);
    digitalWrite(relayPin, HIGH);
    delay(offTime);

    
    //START MOTOR
    analogWrite(ENA, 255);
    analogWrite(ENB, 255);

    Serial.println("starting motor!");
  //   Map speed (0–255) → delay (ms between steps)
  //   Sets speed --> Restate and change "speedvalue" to change the speed
    stepDelayMs = map(speedValue, 0, 255, 10, 1);

    unsigned long startTime = millis();
  //  while (millis() - startTime < (unsigned long)runSeconds * 1000UL) {
  //    if (direction == 'F') stepForward();
  //    else stepBackward();
  //  }

    speedValue = 200;
    while (millis() - startTime < (unsigned long)2 * 1000UL) {
      Serial.println("starting forward");
      stepForward();
    }

    startTime = millis();
    while (millis() - startTime < (unsigned long)2 * 1000UL) {
      Serial.println("starting backward");
      stepBackward();
    }

    speedValue = 150;
    stepDelayMs = map(speedValue, 0, 255, 10, 1);
    startTime = millis();  
    while (millis() - startTime < (unsigned long)2 * 1000UL) {
      Serial.println("starting forward 2");
      stepForward();
    }

    startTime = millis();
    while (millis() - startTime < (unsigned long)2 * 1000UL) {
      Serial.println("starting backward 2");
      stepBackward();
    }

    stopStepper();

    while (player.playingMusic) {
      delay(1000);
    }

    
  }
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
  StaticJsonDocument<256> doc;
  doc["id"] = deviceName;
  doc["ip"] = WiFi.localIP().toString();
  doc["mac"] = WiFi.macAddress();
  doc["type"] = "broadcast";

  char buffer[256];
  size_t n = serializeJson(doc, buffer);

  IPAddress broadcastIP = WiFi.localIP();
  broadcastIP[3] = 255;
  udp.beginPacket(broadcastIP, udpPort);
  udp.write((uint8_t*)buffer, n);
  udp.endPacket();

  Serial.println("Broadcast sent: " + String(buffer));
}
