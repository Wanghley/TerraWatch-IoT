#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <WiFi.h>
#include <HTTPClient.h>

class WiFiManager {
public:
    // Constructor
    WiFiManager();
    
    // Initialize WiFi with credentials
    // For open networks, pass empty string "" for password
    bool begin(const char* ssid, const char* password = "", uint32_t timeoutMs = 15000);
    
    // Check if connected
    bool isConnected();
    
    // Reconnect if disconnected
    bool reconnect(uint32_t timeoutMs = 10000);
    
    // Disconnect from WiFi
    void disconnect();
    
    // Send JSON POST request
    bool sendJSON(const char* serverUrl, const String& jsonPayload);
    
    // Get connection info
    void printConnectionInfo();
    
    // Get signal strength
    int getRSSI();

private:
    String _ssid;
    String _password;
    bool _isOpen;
    
    bool connectToNetwork(uint32_t timeoutMs);
    const char* getStatusString(wl_status_t status);
    void printDebugInfo(wl_status_t status);
};

#endif // WIFI_MANAGER_H