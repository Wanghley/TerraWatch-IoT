#include <SPI.h>
#include <SD.h>
#include <Adafruit_VS1053.h>
#include <Arduino.h>

// === PIN DEFINITIONS FOR MEGA ===
// Stepper L298N (Moved to Digital Pins)
#define ENA 22
#define IN1 23
#define IN2 24
#define ENB 25
#define IN3 26
#define IN4 27

// Relay
const int relayPin = 28;
const bool RELAY_ACTIVE_LOW = true; 

// VS1053 & SD (MUST USE MEGA HARDWARE SPI: 50, 51, 52)
#define VS1053_RST   -1
#define VS1053_CS    40  
#define VS1053_DCS   41  
#define VS1053_DREQ  2   // Interrupt pin on Mega
#define SD_CS        42  

// Create player object
Adafruit_VS1053_FilePlayer player(VS1053_RST, VS1053_CS, VS1053_DCS, VS1053_DREQ, SD_CS);

const char *tracks[] = { "/bear.mp3", "/monster.mp3", "/eagle.mp3", "/owl.mp3", "/dog.mp3" };
const size_t TRACK_COUNT = 5;

// Variables
int speedValue = 255; 
int stepsSeq[4][4] = { {1, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 1} };
int stepDelayMs = 5;

// ==========================================
// === FORWARD DECLARATIONS (THE FIX) ===
// ==========================================
void deterrent();
void lightFlicker(int onDuration, int offDuration);
void motorCall();
void runDirectionFor(unsigned long durationMs, bool forward, int speed);
void stepForward();
void stepBackward();
void setStep(int pins[4]);

// ==========================================
// === SETUP ===
// ==========================================
void setup() {
  Serial.begin(115200);   // Debugging to PC
  Serial1.begin(115200);  // Connection to ESP8266 (Pins 19 RX, 18 TX)
  
  Serial.println("MEGA Starting...");

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, RELAY_ACTIVE_LOW ? HIGH : LOW); // Off initially
  
  // Motor Pins
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Initialize VS1053
  if (!player.begin()) {
    Serial.println("VS1053 not found!");
    // Don't freeze, just continue so motors might work
  } else {
    Serial.println("VS1053 OK");
    if (!SD.begin(SD_CS)) Serial.println("SD failed!");
    player.setVolume(0, 0); 
    player.useInterrupt(VS1053_DREQ);
  }

  randomSeed(analogRead(A0)); 
}

// ==========================================
// === LOOP ===
// ==========================================
void loop() {
  // Listen for commands from ESP8266
  if (Serial1.available()) {
    String cmd = Serial1.readStringUntil('\n');
    cmd.trim();
    
    Serial.print("Received: "); Serial.println(cmd);
    
    if (cmd == "CMD_DETER") {
      deterrent();
    } else if (cmd == "CMD_LIGHT") {
      lightFlicker(8000, 0);
    }
  }
}

// ==========================================
// === LOGIC FUNCTIONS ===
// ==========================================

void deterrent(){
  Serial.println("Running Deterrent...");
  
  if (!player.playingMusic) {
     int track = random(0, TRACK_COUNT);
     Serial.print("Playing: "); Serial.println(tracks[track]);
     player.startPlayingFile(tracks[track]);
  }

  // Flash lights
  for (int i=0; i<5; i++) {
    lightFlicker(random(200, 2000), random(200, 3000));
  }
      
  // Run Motors
  analogWrite(ENA, 255);
  analogWrite(ENB, 255);
  motorCall();

  if(player.playingMusic) player.stopPlaying();
  Serial.println("Deterrent Done");
}

void lightFlicker(int onDuration, int offDuration){
  int onState = RELAY_ACTIVE_LOW ? LOW : HIGH;
  int offState = RELAY_ACTIVE_LOW ? HIGH : LOW;
  
  digitalWrite(relayPin, onState);
  delay(onDuration);
  digitalWrite(relayPin, offState);
  delay(offDuration);
}

void motorCall() {
  int choice = random(0, 5); 
  switch (choice) {
    case 0: runDirectionFor(4000UL, true, 180); delay(333); break;
    case 1: runDirectionFor(2000UL, true, 160); delay(333); runDirectionFor(2000UL, false, 120); delay(333); break;
    case 2: runDirectionFor(1000UL, true, 140); delay(333); runDirectionFor(1000UL, false, 140); delay(333); 
            runDirectionFor(1000UL, true, 90); delay(333); runDirectionFor(1000UL, false, 90); break;
    case 3: runDirectionFor(1000UL, true, 80); delay(333); runDirectionFor(1000UL, true, 140); delay(333); 
            runDirectionFor(1000UL, false, 140); delay(333); runDirectionFor(1000UL, false, 80); delay(333); break;
    case 4: for (int i = 0; i < 8; ++i) { runDirectionFor(500UL, (i%2==0), (i%2==0)?180:120); delay(333); } break;
  }
}

void runDirectionFor(unsigned long durationMs, bool forward, int speed) {
  if (speed > 180) speed = 180;
  if (speed < 0) speed = 0;
  stepDelayMs = map(speed, 0, 255, 10, 1);

  unsigned long startTime = millis();
  while (millis() - startTime < durationMs) {
    if (forward) stepForward(); else stepBackward();
  }
}

void stepForward() {
  for (int i = 0; i < 4; i++) { setStep(stepsSeq[i]); delay(stepDelayMs); }
}

void stepBackward() {
  for (int i = 3; i >= 0; i--) { setStep(stepsSeq[i]); delay(stepDelayMs); }
}

void setStep(int pins[4]) {
  digitalWrite(IN1, pins[0]); digitalWrite(IN2, pins[1]);
  digitalWrite(IN3, pins[2]); digitalWrite(IN4, pins[3]);
}