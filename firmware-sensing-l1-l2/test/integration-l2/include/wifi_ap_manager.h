#ifndef WIFI_AP_MANAGER_H
#define WIFI_AP_MANAGER_H

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

class WiFiAPManager {
public:
    WiFiAPManager();

    // Start soft AP. password = "" for open network.
    void begin(const char* ssid,
               const char* password = "",
               uint8_t channel = 1,
               bool hidden = false,
               uint8_t maxConnections = 4);

    // Print AP info to Serial
    void printAPInfo();

    // Send JSON to an HTTP endpoint (e.g. "http://192.168.4.2/api/detection")
    // Returns true on HTTP 2xx, false otherwise
    bool sendJSON(const char* serverUrl, const String& jsonPayload, uint16_t timeoutMs = 5000);

    // Stop AP / disconnect
    void disconnect();

private:
    // helper to ensure HTTPClient is safe to use
    bool isAPRunning();
};

#endif // WIFI_AP_MANAGER_H
