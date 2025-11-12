#include "wifi_manager.h"
#include <Arduino.h>

// Initialize the static variable
volatile bool WifiManager::_wifiConnected = false;
static WifiManager* instancePtr = nullptr;

WifiManager::WifiManager(const char* ssid, const char* password, bool debug, unsigned int updPort, const char* targetId, IPAddress staticIP)
  : _ssid(ssid), _password(password), _debug(debug), _udpPort(udpPort), _targetId(targetId), _staticIP(staticIP)  {
    instancePtr = this; // store pointer for static handler
}


void WifiManager::handleWiFiEvent(WiFiEvent_t event) {
    if (!instancePtr) return;

    if (instancePtr->_debug) {
        switch (event) {
            case ARDUINO_EVENT_WIFI_STA_START:
                Serial.println("WiFi STA Started...");
                break;
            case ARDUINO_EVENT_WIFI_STA_GOT_IP:
                Serial.print("WiFi connected! IP: ");
                Serial.println(WiFi.localIP());
                identifyIP();
                break;
            case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
                Serial.println("WiFi disconnected.");
                break;
            default: break;
        }
    }

    if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
        _wifiConnected = true;
    } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
        _wifiConnected = false;
    }
}


void WifiManager::connect() {
  _wifiConnected = false;
  WiFi.mode(WIFI_STA);
  WiFi.onEvent(handleWiFiEvent); // Register our static event handler
  WiFi.begin(_ssid, _password);

  if (_debug) {
    Serial.print("Connecting to WiFi SSID: ");
    Serial.println(_ssid);
  }

  unsigned long start = millis();
  while (!_wifiConnected && millis() - start < 10000) { // Wait 10s
    delay(300);
    Serial.print(".");
  }

  if (_wifiConnected) {
    if (_debug) {
      Serial.println("\n✅ WiFi connected successfully!");
    }
    
    // THIS IS THE KEY: Enable Wi-Fi modem sleep.
    // This keeps the connection alive during light sleep.
    WiFi.setSleep(true);
    
  } else {
    if (_debug) {
      Serial.println("\n⚠️ WiFi connection failed on startup.");
    }
  }
}

bool WifiManager::isConnected() {
  return (WiFi.status() == WL_CONNECTED);
  // We also check the official status, just in case.
}

void WifiManager::disconnect() {
  WiFi.disconnect(true); // true = also erase credentials
  _wifiConnected = false;
  if (_debug) {
    Serial.println("WiFi disconnected.");
  }
}

void WifiManager::reconnect() {
  if (_debug) {
    Serial.println("Reconnecting to WiFi...");
  }
  WiFi.reconnect();
}

void WifiManager::identifyIP() {
  udp.begin(udpPort);
  Serial.printf("Listening for UDP JSON packets on port %d...\n", udpPort);
  while(1){
    int packetSize = udp.parsePacket();
    if (packetSize) {
      int len = udp.read(incomingPacket, sizeof(incomingPacket) - 1);
      if (len > 0) incomingPacket[len] = '\0';

      Serial.printf("\nReceived: %s\n", incomingPacket);

      StaticJsonDocument<512> doc;
      DeserializationError error = deserializeJson(doc, incomingPacket);
      if (error) {
        Serial.println("JSON parse failed");
        return;
      }

      String id = doc["id"];
      String ip = doc["ip"];
      String mac = doc["mac"];
      String type = doc["type"];

      Serial.printf("Device ID: %s | IP: %s | MAC: %s | Type: %s\n",
                    id.c_str(), ip.c_str(), mac.c_str(), type.c_str());

      // Check if this is the one we want to stop
      if (id == targetId && type == "broadcast") {
        Serial.println("Matched target! Sending STOP command...");

        StaticJsonDocument<128> reply;
        reply["type"] = "stop";
        reply["target"] = id;

        char buffer[128];
        size_t n = serializeJson(reply, buffer);

        udp.beginPacket(udp.remoteIP(), udp.remotePort());
        udp.write((uint8_t*)buffer, n);
        udp.endPacket();

        Serial.println("STOP message sent.");
        break;
      }
    }
  }
}

bool WifiManager::triggerDeterrenceSystem(float probability, float threshold, const char* modelVersion, const char* deviceID) {
  if (!isConnected()) {
    if (_debug) {
      Serial.println("Not connected to WiFi, cannot trigger deterrence system.");
    }
    return false;
  }

  // prepare json payload
  String jsonPayload = "{";
  jsonPayload += "\"action\":\"trigger_deterrence\",";
  jsonPayload += "\"activated\":true,";
  jsonPayload += "\"timestamp\":" + String((uint32_t)time(nullptr)) + ",";
  jsonPayload += "\"trigger_reason\":\"probability_threshold\",";
  jsonPayload += "\"probability\":" + String(probability, 6) + ",";
  jsonPayload += "\"threshold\":" + String(threshold, 6) + ",";
  jsonPayload += "\"model_version\":\"" + String(modelVersion) + "\",";
  jsonPayload += "\"device_id\":\"" + String(deviceID) + "\"";
  jsonPayload += "}";

  // send get request with json information to deterrence system
  String url = "/";
  if (_debug) {
    Serial.print("Triggering deterrence system with payload: ");
    Serial.println(jsonPayload);
  }

  if (_wifiClient.connect(_ipAddress, 80)) {
    _wifiClient.println("POST " + url + " HTTP/1.1");
    _wifiClient.println("Host: " + _ipAddress.toString());
    _wifiClient.println("Content-Type: application/json");
    _wifiClient.println("Content-Length: " + String(jsonPayload.length()));
    _wifiClient.println();
    _wifiClient.print(jsonPayload);
    _wifiClient.stop();
    if (_debug) {
      Serial.println("Deterrence system triggered successfully.");
    }
    return true;
  } else {
    if (_debug) {
      Serial.println("Failed to connect to deterrence system");
    }
    return false;
  }