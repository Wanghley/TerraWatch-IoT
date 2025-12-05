#include <SPI.h>
#include <SD.h>
#include <Adafruit_VS1053.h>
#include <Arduino.h>

// === PIN DEFINITIONS ===
#define ENA 22

// REVERTED TO STANDARD LOGICAL ORDER
// Ensure your Motor Wires are paired correctly on the board (Pair A -> Out1/2, Pair B -> Out3/4)
#define IN1 23
#define IN2 24
#define ENB 25
#define IN3 26
#define IN4 27

// Relay
const int relayPin = 28;
const bool RELAY_ACTIVE_LOW = false; 

// VS1053 (Hardware SPI: 50, 51, 52)
#define VS1053_RST   -1
#define VS1053_CS    40  
#define VS1053_DCS   41  
#define VS1053_DREQ  2   
#define SD_CS        42  

Adafruit_VS1053_FilePlayer player(VS1053_RST, VS1053_CS, VS1053_DCS, VS1053_DREQ, SD_CS);

const char *tracks[] = { "/bear.mp3", "/monster.mp3", "/eagle.mp3", "/owl.mp3", "/dog.mp3" };
const size_t TRACK_COUNT = 5;

// Variables
int stepsSeq[4][4] = { {1, 0, 1, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 0, 1} };

// INCREASED DELAY: 3ms might be too fast causing slippage/jitter. 
// Try 10ms. If smooth, lower to 5ms.
int stepDelayMs = 50; 

// === FORWARD DECLARATIONS ===
void deterrent();
void runDirectorMode(unsigned long durationMs); 
void setStep(int pins[4]);
void clearSerialBuffer();
void safeDelay(unsigned long ms);

// === SETUP ===
void setup() {
  Serial.begin(115200);   
  Serial1.begin(115200);  
  
  Serial.println("MEGA Starting...");

  // 1. MEGA SPI FIX
  pinMode(53, OUTPUT);
  digitalWrite(53, HIGH);

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, RELAY_ACTIVE_LOW ? HIGH : LOW); 
  
  // 2. MOTOR PINS SETUP
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Turn Motors ON (Full Power)
  digitalWrite(ENA, HIGH); 
  digitalWrite(ENB, HIGH);

  // 3. Audio / SD Setup
  pinMode(VS1053_CS, OUTPUT); digitalWrite(VS1053_CS, HIGH);
  pinMode(VS1053_DCS, OUTPUT); digitalWrite(VS1053_DCS, HIGH);

  if (!SD.begin(SD_CS)) {
    Serial.println("ERROR: SD failed!");
  } else {
    Serial.println("SD OK");
  }

  if (!player.begin()) {
    Serial.println("ERROR: VS1053 not found!");
  } else {
    Serial.println("VS1053 OK");
    player.setVolume(0, 0); 
  }
  randomSeed(analogRead(A0)); 
}

// === LOOP ===
void loop() {
  if (Serial1.available()) {
    String cmd = Serial1.readStringUntil('\n');
    cmd.trim();
    if(cmd.length() > 0) {
        Serial.print("Received: "); Serial.println(cmd);
        if (cmd == "CMD_DETER") {
          deterrent();
        } else if (cmd == "CMD_LIGHT") {
          // Manual light trigger (8 seconds)
          digitalWrite(relayPin, RELAY_ACTIVE_LOW ? LOW : HIGH);
          safeDelay(2000); 
          digitalWrite(relayPin, RELAY_ACTIVE_LOW ? HIGH : LOW);
          clearSerialBuffer(); 
        }
    }
  }
  if (player.playingMusic) player.feedBuffer();
}

// === HELPER: Safe Delay for Manual Light Trigger ===
void safeDelay(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    if (player.playingMusic) player.feedBuffer();
    yield();
  }
}

// === MAIN LOGIC ===
void deterrent(){
  Serial.println("--- START DETERRENT ---");
  
  // 1. Start Music
  if (!player.playingMusic) {
     int track = random(0, TRACK_COUNT);
     Serial.print("Playing: "); Serial.println(tracks[track]);
     if (player.startPlayingFile(tracks[track])) {
       // Fill buffer slightly
       for(int i=0; i<30; i++) { player.feedBuffer(); delay(1); }
     }
  }

  // 2. RUN DIRECTOR MODE (Scarier Logic)
  // Run for 12 seconds
  runDirectorMode(12000); 

  // 3. Stop Everything
  if(player.playingMusic) {
    player.stopPlaying();
  }
  
  // Turn off motors
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  
  // Turn off light
  digitalWrite(relayPin, RELAY_ACTIVE_LOW ? HIGH : LOW);

  clearSerialBuffer();
  Serial.println("--- DONE ---");
}

// === THE DIRECTOR FUNCTION ===
// Cycles through different "Scare Behaviors" dynamically
void runDirectorMode(unsigned long durationMs) {
  unsigned long globalEnd = millis() + durationMs;
  
  // State Machine Variables
  int currentBehavior = 0;
  unsigned long behaviorEnd = 0;
  
  // Motor State
  int stepIndex = 0;
  bool motorDirection = true;
  unsigned long lastStepMicros = 0;
  unsigned long stepInterval = 3000; // 3ms default
  unsigned long lastShakeToggle = 0;

  // Light State
  bool lightState = false;
  unsigned long nextLightToggle = 0;

  Serial.println("DIRECTOR MODE: ACTION!");

  while(millis() < globalEnd) {
    // 1. FEED AUDIO ALWAYS
    if (player.playingMusic) player.feedBuffer();

    // 2. PICK NEW BEHAVIOR if current one finished
    if (millis() > behaviorEnd) {
      currentBehavior = random(0, 4); // Pick one of 4 modes
      unsigned long modeDuration = random(1500, 3500); // 1.5 to 3.5 seconds per mode
      behaviorEnd = millis() + modeDuration;
      
      // Initial Setup for modes
      switch (currentBehavior) {
        case 0: Serial.println("MODE: VIOLENT STROBE"); break;
        case 1: Serial.println("MODE: THE CREEPER"); break;
        case 2: Serial.println("MODE: THE CHASE"); break;
        case 3: Serial.println("MODE: GLITCH"); break;
      }
    }

    // 3. EXECUTE CURRENT BEHAVIOR
    switch (currentBehavior) {
      // --- MODE 0: VIOLENT STROBE & SHAKE (Max Intensity) ---
      case 0: 
        stepInterval = 6000; // Increased from 2500 to 6000 for safety
        // Shake Motor: Switch direction rapidly
        if (millis() - lastShakeToggle > 100) { // Slower shake to prevent stalling
           motorDirection = !motorDirection;
           lastShakeToggle = millis();
        }
        // Strobe Light: Fast 25Hz strobe
        if (millis() > nextLightToggle) {
          lightState = !lightState;
           int s = lightState ? (RELAY_ACTIVE_LOW ? LOW : HIGH) : (RELAY_ACTIVE_LOW ? HIGH : LOW);
           digitalWrite(relayPin, s);
           nextLightToggle = millis() + 40; 
        }
        break;

      // --- MODE 1: THE CREEPER (Slow & Failing Light) ---
      case 1:
        stepInterval = 15000; // 15ms (Slow, creepy crawl)
        motorDirection = true; // One direction
        // Flickering Lightbulb Effect
        if (millis() > nextLightToggle) {
           // Mostly ON, brief OFFs
           lightState = !lightState;
           // If we just turned it ON, keep it ON for longer. If OFF, quick flicker.
           unsigned long wait = lightState ? random(200, 800) : random(20, 100);
           
           int s = lightState ? (RELAY_ACTIVE_LOW ? LOW : HIGH) : (RELAY_ACTIVE_LOW ? HIGH : LOW);
           digitalWrite(relayPin, s);
           nextLightToggle = millis() + wait;
        }
        break;

      // --- MODE 2: THE CHASE (Fast & Bright) ---
      case 2:
        stepInterval = 4000; // 4ms (Fast but safe)
        motorDirection = true; // Run!
        // Light Solid ON (See the monster)
        digitalWrite(relayPin, RELAY_ACTIVE_LOW ? LOW : HIGH);
        break;

      // --- MODE 3: THE GLITCH (Chaos) ---
      case 3:
         // Random Speed changes
         if (random(0, 100) > 90) stepInterval = random(5000, 10000);
         // Random Direction
         if (random(0, 100) > 95) motorDirection = !motorDirection;
         // Random Light
         if (millis() > nextLightToggle) {
            lightState = !lightState;
            int s = lightState ? (RELAY_ACTIVE_LOW ? LOW : HIGH) : (RELAY_ACTIVE_LOW ? HIGH : LOW);
            digitalWrite(relayPin, s);
            nextLightToggle = millis() + random(50, 500);
         }
         break;
    }

    // 4. STEP MOTOR
    if (micros() - lastStepMicros >= stepInterval) {
      lastStepMicros = micros();
      
      if (motorDirection) {
        stepIndex++;
        if (stepIndex > 3) stepIndex = 0;
      } else {
        stepIndex--;
        if (stepIndex < 0) stepIndex = 3;
      }
      setStep(stepsSeq[stepIndex]);
    }
  }
}

void setStep(int pins[4]) {
  digitalWrite(IN1, pins[0]); digitalWrite(IN2, pins[1]);
  digitalWrite(IN3, pins[2]); digitalWrite(IN4, pins[3]);
}

void clearSerialBuffer() {
  while(Serial1.available()) Serial1.read(); 
}