#include <SPI.h>
#include <SD.h>
#include <Adafruit_VS1053.h>

// L298N pin assignments
#define ENA 5      // PWM capable
#define IN1 4      // Digital
#define IN2 7      // Digital
#define ENB 9      // PWM capable
#define IN3 15     // A1 (digital pin 15) - changed from 10 to avoid SD_CS conflict
#define IN4 14     // A0 (digital pin 14)

// === PIN CONFIGURATION for Arduino UNO ===
#define VS1053_RST  -1   // not wired
#define VS1053_CS    8   // VS1053 XCS (adjusted for UNO)
#define VS1053_DCS   6   // VS1053 XDCS (adjusted for UNO)
#define VS1053_DREQ  3   // VS1053 DREQ (adjusted for UNO)
#define SD_CS       10   // SD card CS (adjusted for UNO)

// === Input pins from ESP32 ===
const int lightTriggerPin = 2;      // Pin that goes HIGH when light trigger detected (interrupt capable)
const int deterrentTriggerPin = 16; // A2 (digital pin 16) - changed from 11 to avoid SPI MOSI conflict

// === File to play (use absolute path) ===
const char *tracks[] = {
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

const int relayPin = 17;  // A3 (digital pin 17) - changed from 12 to avoid SPI MISO conflict
const bool RELAY_ACTIVE_LOW = true;  // change to false if your relay is active HIGH

// Pin state tracking for edge detection
bool lightTriggerLastState = LOW;
bool deterrentTriggerLastState = LOW;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Starting MP3 loop player...");

  // Initialize SPI & SD
  SPI.begin();
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);
  if (!SD.begin(SD_CS)) {
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
  
  // Input pins from ESP32 (using INPUT since ESP32 will drive HIGH/LOW)
  pinMode(lightTriggerPin, INPUT);
  pinMode(deterrentTriggerPin, INPUT);

  // Motor control pins
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Set moderate volume (0 loudest)
  player.setVolume(0, 0);
  // Enable interrupt-driven playback
  #if defined(VS1053_FILEPLAYER_PIN_INT)
    player.useInterrupt(VS1053_FILEPLAYER_PIN_INT);
  #else
    player.useInterrupt(VS1053_DREQ); 
  #endif

  randomSeed(analogRead(0));  // randomize using analog input

  Serial.println("Initialization complete.");
}

void loop() {
  // Monitor trigger pins for rising edge (LOW -> HIGH)
  bool lightTriggerCurrentState = digitalRead(lightTriggerPin);
  bool deterrentTriggerCurrentState = digitalRead(deterrentTriggerPin);
  
  bool lightTriggerRising = (lightTriggerCurrentState == HIGH && lightTriggerLastState == LOW);
  bool deterrentTriggerRising = (deterrentTriggerCurrentState == HIGH && deterrentTriggerLastState == LOW);
  
  // Handle light trigger
  if (lightTriggerRising) {
    Serial.println("Light trigger detected - keeping lights on for 8 seconds");
    lightFlicker(8000, 0);
    Serial.println("Lights turned off");
  }
  // Handle deterrent trigger
  else if (deterrentTriggerRising) {
    Serial.println("Deterrent trigger detected - running full deterrent sequence");
    deterrent();
  }
  
  // Update pin states for next iteration
  lightTriggerLastState = lightTriggerCurrentState;
  deterrentTriggerLastState = deterrentTriggerCurrentState;
  
  delay(10);
}

void deterrent(){
  if (!player.playingMusic) {
    const char* MP3_FILE = tracks[pickRandomTrack()];
    Serial.println("Starting playback");
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

int pickRandomTrack() {
  randomSeed(analogRead(0));
  return random(0, TRACK_COUNT);
}

void lightFlicker(int onDuration, int offDuration){
  // For active-low relay: LOW = ON, HIGH = OFF
  // For active-high relay: HIGH = ON, LOW = OFF
  int onState = RELAY_ACTIVE_LOW ? LOW : HIGH;
  int offState = RELAY_ACTIVE_LOW ? HIGH : LOW;
  
  digitalWrite(relayPin, onState);
  delay(onDuration);
  digitalWrite(relayPin, offState);
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
      // Preset 4: short pulses: 4 alternating 0.5s pulses
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

