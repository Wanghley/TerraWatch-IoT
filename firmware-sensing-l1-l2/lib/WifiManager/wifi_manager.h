#pragma once // Prevents file from being included multiple times

#include <WiFi.h>
#include <WiFiUDP.h>
#include <ArduinoJSON.h>

class WifiManager {
public:
    // Constructor: Takes SSID and password when created
    WifiManager(const char* ssid, const char* password, bool debug = false, IPAddress staticIP = IPAddress(0,0,0,0));

    // Main connection function (blocking)
    void connect();

    // Quick check to see if we are still connected
    bool isConnected();

    // Disconnect from Wi-Fi
    void disconnect();

    // Reconnect to Wi-Fi
    void reconnect();

    // Identify IPAddress of Deter ESP32
    void identifyIP();

    // Trigger to send request to deterrence system wifi client
    bool triggerDeterrenceSystem(float probability = 1.0, float threshold = 0.5, const char* modelVersion = "1.0", const char* deviceID = "AGRONAUTS_L1_L2");

private:
    const char* _ssid;
    const char* _password;
    const char* targetId;
    bool _debug;
    unsigned int udpPort;

    char incomingPacket[512];
    
    // create wifi client
    WiFiClient _wifiClient;
    IPAddress _ipAddress;
    WiFiUDP udp;
    // This is the static event handler required by the ESP-IDF
    static void handleWiFiEvent(WiFiEvent_t event);

    // A static flag to be set by the event handler
    static volatile bool _wifiConnected;
};