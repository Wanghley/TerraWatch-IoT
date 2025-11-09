#include "wifi_manager.h"
#include <Arduino.h>

// Initialize the static variable
volatile bool WifiManager::_wifiConnected = false;
static WifiManager* instancePtr = nullptr;

WifiManager::WifiManager(const char* ssid, const char* password, bool debug)
  : _ssid(ssid), _password(password), _debug(debug) {
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