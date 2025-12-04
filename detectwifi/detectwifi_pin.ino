// ESP32 sensor monitoring - outputs HIGH signals to Arduino UNO
// No WiFi, UDP, or HTTP - direct pin communication

const int lightPin = 14;           // Input: sensor pin for light trigger
const int orangepipin = 33;        // Input: sensor pin for deterrent trigger

// Output pins connected to Arduino UNO
const int lightTriggerOutPin = 26;      // Output: goes HIGH when light trigger detected
const int deterrentTriggerOutPin = 27;  // Output: goes HIGH when deterrent trigger detected

// Pin state tracking for edge detection
bool lightLastPinState = LOW;
bool orangepiLastPinState = LOW;

void setup() {
  Serial.begin(115200);
  
  // Configure input pins
  pinMode(lightPin, INPUT_PULLDOWN);
  pinMode(orangepipin, INPUT_PULLDOWN);
  
  // Configure output pins (connect to Arduino UNO)
  pinMode(lightTriggerOutPin, OUTPUT);
  pinMode(deterrentTriggerOutPin, OUTPUT);
  digitalWrite(lightTriggerOutPin, LOW);
  digitalWrite(deterrentTriggerOutPin, LOW);
  
  Serial.println("ESP32 Sensor Monitor Ready");
  Serial.println("Monitoring pins for triggers...");
}

void loop() {
  // Monitor sensor pins for rising edge (LOW -> HIGH)
  bool lightCurrentState = digitalRead(lightPin);
  bool orangepiCurrentState = digitalRead(orangepipin);
  
  bool lightPinRising = (lightCurrentState == HIGH && lightLastPinState == LOW);
  bool orangepiPinRising = (orangepiCurrentState == HIGH && orangepiLastPinState == LOW);
  
  // Handle light trigger - set output pin HIGH
  if (lightPinRising) {
    Serial.println("Light pin went HIGH -> Setting light trigger output HIGH");
    digitalWrite(lightTriggerOutPin, HIGH);
    delay(100); // Brief pulse to trigger Arduino UNO
    digitalWrite(lightTriggerOutPin, LOW);
    Serial.println("Light trigger signal sent");
  }
  // Handle deterrent trigger - set output pin HIGH
  else if (orangepiPinRising) {
    Serial.println("OrangePi pin went HIGH -> Setting deterrent trigger output HIGH");
    digitalWrite(deterrentTriggerOutPin, HIGH);
    delay(100); // Brief pulse to trigger Arduino UNO
    digitalWrite(deterrentTriggerOutPin, LOW);
    Serial.println("Deterrent trigger signal sent");
  }
  
  // Update pin states for next iteration
  lightLastPinState = lightCurrentState;
  orangepiLastPinState = orangepiCurrentState;
  
  delay(10);
}

