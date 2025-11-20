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

  // Begin UDP listening for STOP message
  udp.begin(listenPort);
  Serial.print("UDP listening on port ");
  Serial.println(listenPort);

  while(true){
    // Broadcast every 5 seconds
    if (millis() - lastBroadcast > 5000) {
      sendBroadcast();
      lastBroadcast = millis();
    }

    // Check if any message is received
    int packetSize = udp.parsePacket();
    if (packetSize) {
      char buffer[256];
      int len = udp.read(buffer, 255);
      if (len > 0) buffer[len] = '\0';

      Serial.print("Received: ");
      Serial.println(buffer);

      if (strcmp(buffer, STOP_MSG) == 0) {
        Serial.println("STOP message received! Halting broadcast.");
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

  Serial.println("Initialization complete.");
}

void loop() {
  // When finished, restart

  Serial.println("Looking for client connection.");
  WiFiClient client = server.available();
  if (!client) return;

  Serial.println("Client connected");

  // Wait for header
  while (!client.available()) delay(1);

  
  String req = client.readStringUntil('\r');
  Serial.println("Request:");
  Serial.println(req);

  // Read full header until blank line
  while (client.available()) {
    String line = client.readStringUntil('\r');
    if (line == "\n" || line == "\r\n") break;
  }

  // -----------------------------------
  //   PROCESS POST REQUEST & TURN ON DETER
  // -----------------------------------
  if (req.startsWith("POST")) {
    Serial.println("POST request detected");
    deterrent();
  }

    // -----------------------------------
    //   SEND HTTP RESPONSE
    // -----------------------------------
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/plain");
    client.println("Connection: close");
    client.println();
    client.println("ACK from Sender ESP32");

  delay(10);
  client.stop();
  Serial.println("Client disconnected");
  
}

void deterrent(){
  if (!player.playingMusic) {
        MP3_FILE = tracks[pickRandomTrack()]
        Serial.println("Connected to client Starting playback");
        player.startPlayingFile(MP3_FILE);
      }


      for (int i=0; i<5; i++) {
        // Random ON duration between 200ms and 2000ms
        int onTime = random(200, 2000);

        // Random OFF duration between 200ms and 3000ms
        int offTime = random(200, 3000);

        digitalWrite(relayPin, HIGH);
        delay(onTime);
        digitalWrite(relayPin, LOW);
        delay(offTime);
      }

      
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
      Serial.println("starting forward");
      while (millis() - startTime < (unsigned long)1 * 1000UL) {
        stepForward();
      }

      startTime = millis();
      Serial.println("starting backward");
      while (millis() - startTime < (unsigned long)1 * 1000UL) {
        Serial.println("starting backward");
        stepBackward();
      }

      speedValue = 150;
      stepDelayMs = map(speedValue, 0, 255, 10, 1);
      startTime = millis();  
      while (millis() - startTime < (unsigned long)1 * 1000UL) {
        Serial.println("starting forward 2");
        stepForward();
      }

      startTime = millis();
      while (millis() - startTime < (unsigned long)1 * 1000UL) {
        Serial.println("starting backward 2");
        stepBackward();
      }

      stopStepper();

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

int pickRandomTrack() {
  uint32_t seed = esp_random();
  randomSeed(seed)
  return random(0, TRACK_COUNT)
}

