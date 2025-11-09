#pragma once // Prevents file from being included multiple times

#include <WiFi.h>

class WifiManager {
public:
    // Constructor: Takes SSID and password when created
    WifiManager(const char* ssid, const char* password, bool debug = false);

    // Main connection function (blocking)
    void connect();

    // Quick check to see if we are still connected
    bool isConnected();

    // Disconnect from Wi-Fi
    void disconnect();

    // Reconnect to Wi-Fi
    void reconnect();

private:
    const char* _ssid;
    const char* _password;
    bool _debug;

    // This is the static event handler required by the ESP-IDF
    static void handleWiFiEvent(WiFiEvent_t event);

    // A static flag to be set by the event handler
    static volatile bool _wifiConnected;
};