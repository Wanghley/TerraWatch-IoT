#include "wifi_manager.h"

WiFiManager::WiFiManager() : _isOpen(false) {
}

bool WiFiManager::begin(const char* ssid, const char* password, uint32_t timeoutMs) {
    _ssid = String(ssid);
    _password = String(password);
    _isOpen = (strlen(password) == 0);
    
    Serial.println("\n=== WiFi Manager Initializing ===");
    Serial.printf("SSID: '%s' (length: %d)\n", ssid, strlen(ssid));
    Serial.printf("Password length: %d\n", strlen(password));
    Serial.printf("Network Type: %s\n", _isOpen ? "Open" : "Secured");
    
    // Completely reset WiFi subsystem
    WiFi.disconnect(true);
    delay(1000);
    WiFi.mode(WIFI_OFF);
    delay(1000);
    
    // Enable STA mode with proper configuration
    WiFi.mode(WIFI_STA);
    delay(1000);
    
    // Critical: Disable persistent storage (can cause connection issues)
    WiFi.persistent(false);
    
    // Disable auto-reconnect during initial setup
    WiFi.setAutoReconnect(false);
    
    // Set hostname (optional but can help with some routers)
    WiFi.setHostname("ESP32-Detector");
    
    // Disable power saving mode for more reliable connection
    WiFi.setSleep(false);
    
    // Set explicit WiFi channel and BSSID settings off
    WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
    
    // Increase WiFi TX power for better connection stability
    WiFi.setTxPower(WIFI_POWER_19_5dBm);
    
    // Try multiple connection attempts with different strategies
    for (int attempt = 1; attempt <= 3; attempt++) {
        Serial.printf("\n--- Connection Attempt %d/3 ---\n", attempt);
        
        if (connectToNetwork(timeoutMs)) {
            return true;
        }
        
        if (attempt < 3) {
            Serial.println("Retrying with full reset...");
            WiFi.disconnect(true);
            delay(2000);
        }
    }
    
    Serial.println("\n✗ All connection attempts failed!");
    return false;
}

bool WiFiManager::connectToNetwork(uint32_t timeoutMs) {
    Serial.println("Connecting to WiFi...");
    
    // Scan for the network first to verify it exists
    Serial.println("Scanning for networks...");
    int n = WiFi.scanNetworks();
    bool networkFound = false;
    int8_t targetRSSI = -100;
    uint8_t targetChannel = 0;
    uint8_t* targetBSSID = nullptr;
    wifi_auth_mode_t targetEncryption = WIFI_AUTH_OPEN;
    
    Serial.printf("Found %d networks:\n", n);
    for (int i = 0; i < n; i++) {
        String currentSSID = WiFi.SSID(i);
        Serial.printf("  %d: '%s' (RSSI: %d, Ch: %d, Enc: %d)\n", 
                     i + 1, 
                     currentSSID.c_str(), 
                     WiFi.RSSI(i),
                     WiFi.channel(i),
                     WiFi.encryptionType(i));
        
        if (currentSSID == _ssid) {
            networkFound = true;
            targetRSSI = WiFi.RSSI(i);
            targetChannel = WiFi.channel(i);
            targetEncryption = WiFi.encryptionType(i);
            Serial.printf("  -> Target network found!\n");
            Serial.printf("     RSSI: %d dBm, Channel: %d, Encryption: %d\n", 
                         targetRSSI, targetChannel, targetEncryption);
        }
    }
    
    if (!networkFound) {
        Serial.println("✗ Target network not found in scan!");
        Serial.println("  Check: 1) SSID is correct 2) Router is on 2.4GHz 3) Signal strength");
        return false;
    }
    
    if (targetRSSI < -80) {
        Serial.printf("⚠ WARNING: Weak signal (%d dBm). Move closer to router.\n", targetRSSI);
    }
    
    // Check encryption type matches expectations
    if (_isOpen && targetEncryption != WIFI_AUTH_OPEN) {
        Serial.println("⚠ WARNING: Network is secured but no password provided!");
        Serial.println("  Provide a password or check if network is truly open.");
    }
    
    if (!_isOpen && targetEncryption == WIFI_AUTH_OPEN) {
        Serial.println("⚠ WARNING: Network is open but password was provided!");
        Serial.println("  Try using empty string \"\" for password.");
    }
    
    // IMPORTANT: Give WiFi stack time to settle after scan
    delay(100);
    
    // Begin connection
    Serial.println("\nInitiating connection...");
    Serial.printf("Attempting: SSID='%s', Password length=%s\n", 
                 _ssid.c_str(), _password.c_str());
    
    // Use WiFiMulti approach for more reliable connection
    wl_status_t status;
    
    if (_isOpen) {
        // For open networks
        status = WiFi.begin(_ssid.c_str());
    } else {
        // For secured networks - try with explicit null termination
        status = WiFi.begin(_ssid.c_str(), _password.c_str());
    }
    
    // Enable auto-reconnect after begin
    WiFi.setAutoReconnect(true);
    
    Serial.printf("Begin returned status: %s (%d)\n", getStatusString(status), status);
    
    // Wait for connection with detailed status reporting
    unsigned long startAttempt = millis();
    wl_status_t lastStatus = status;
    int dotCount = 0;
    
    while (WiFi.status() != WL_CONNECTED && 
           millis() - startAttempt < timeoutMs) {
        
        wl_status_t currentStatus = WiFi.status();
        
        // Print status changes
        if (currentStatus != lastStatus) {
            Serial.printf("\nStatus: %s (%d)\n", getStatusString(currentStatus), currentStatus);
            lastStatus = currentStatus;
            dotCount = 0;
            
            // Check for immediate failures
            if (currentStatus == WL_CONNECT_FAILED) {
                Serial.println("Connection failed - likely password issue!");
                delay(1000); // Give it a moment before giving up
            }
            
            if (currentStatus == WL_NO_SSID_AVAIL) {
                Serial.println("SSID disappeared - check router!");
                break;
            }
        }
        
        Serial.print(".");
        dotCount++;
        if (dotCount >= 40) {
            Serial.println();
            dotCount = 0;
        }
        delay(500);
    }
    Serial.println();
    
    wl_status_t finalStatus = WiFi.status();
    
    if (finalStatus == WL_CONNECTED) {
        Serial.println("✓ WiFi Connected Successfully!");
        printConnectionInfo();
        return true;
    } else {
        Serial.println("✗ WiFi Connection Failed!");
        Serial.printf("Final Status: %s (%d)\n", getStatusString(finalStatus), finalStatus);
        printDebugInfo(finalStatus);
        return false;
    }
}

const char* WiFiManager::getStatusString(wl_status_t status) {
    switch(status) {
        case WL_IDLE_STATUS: return "IDLE";
        case WL_NO_SSID_AVAIL: return "NO_SSID_AVAILABLE";
        case WL_SCAN_COMPLETED: return "SCAN_COMPLETED";
        case WL_CONNECTED: return "CONNECTED";
        case WL_CONNECT_FAILED: return "CONNECT_FAILED";
        case WL_CONNECTION_LOST: return "CONNECTION_LOST";
        case WL_DISCONNECTED: return "DISCONNECTED";
        default: return "UNKNOWN";
    }
}

void WiFiManager::printDebugInfo(wl_status_t status) {
    Serial.println("\n--- Troubleshooting Guide ---");
    
    switch(status) {
        case WL_NO_SSID_AVAIL:  // Status 1
            Serial.println("ERROR: SSID not found (Status 1)");
            Serial.println("Solutions:");
            Serial.println("  1. Check SSID spelling (case-sensitive!)");
            Serial.println("  2. Ensure router is on 2.4GHz (ESP32 doesn't support 5GHz)");
            Serial.println("  3. Check if router is broadcasting SSID (not hidden)");
            Serial.println("  4. Move closer to the router");
            break;
            
        case WL_CONNECT_FAILED:  // Status 5
            Serial.println("ERROR: Authentication failed (Status 5)");
            Serial.println("Solutions:");
            Serial.println("  1. Check password is correct");
            Serial.println("  2. For open networks, use empty string \"\"");
            Serial.println("  3. Check router security type (WPA2 recommended)");
            Serial.println("  4. Disable MAC filtering on router");
            Serial.println("  5. Check if router has client limit");
            Serial.println("  6. Try rebooting the router");
            break;
            
        case WL_CONNECTION_LOST:
            Serial.println("ERROR: Connection lost");
            Serial.println("Check signal strength and interference");
            break;
            
        default:
            Serial.printf("Status code: %d\n", status);
            Serial.println("Check WiFi credentials and router settings");
            break;
    }
    
    Serial.println("\nESP32 WiFi Info:");
    Serial.printf("  MAC Address: %s\n", WiFi.macAddress().c_str());
    Serial.printf("  Hostname: %s\n", WiFi.getHostname());
    Serial.println("----------------------------\n");
}

bool WiFiManager::isConnected() {
    return (WiFi.status() == WL_CONNECTED);
}

bool WiFiManager::reconnect(uint32_t timeoutMs) {
    if (isConnected()) {
        return true;
    }
    
    Serial.println("WiFi disconnected. Reconnecting...");
    WiFi.disconnect(true);
    delay(1000);
    
    return connectToNetwork(timeoutMs);
}

void WiFiManager::disconnect() {
    WiFi.disconnect(true);
    Serial.println("WiFi disconnected.");
}

bool WiFiManager::sendJSON(const char* serverUrl, const String& jsonPayload) {
    if (!isConnected()) {
        Serial.println("Not connected to WiFi. Attempting reconnect...");
        if (!reconnect()) {
            Serial.println("Reconnection failed. Cannot send data.");
            return false;
        }
    }
    
    HTTPClient http;
    bool success = false;
    
    Serial.println("\n=== Sending HTTP POST ===");
    Serial.printf("URL: %s\n", serverUrl);
    Serial.printf("Payload: %s\n", jsonPayload.c_str());
    
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);
    
    int httpResponseCode = http.POST(jsonPayload);
    
    if (httpResponseCode > 0) {
        Serial.printf("✓ HTTP Response code: %d\n", httpResponseCode);
        String response = http.getString();
        Serial.printf("Response: %s\n", response.c_str());
        success = (httpResponseCode >= 200 && httpResponseCode < 300);
    } else {
        Serial.printf("✗ HTTP Error: %s\n", http.errorToString(httpResponseCode).c_str());
        success = false;
    }
    
    http.end();
    return success;
}

void WiFiManager::printConnectionInfo() {
    Serial.println("\n--- Connection Info ---");
    Serial.printf("✓ SSID: %s\n", WiFi.SSID().c_str());
    Serial.printf("✓ IP Address: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("✓ Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    Serial.printf("✓ Subnet: %s\n", WiFi.subnetMask().toString().c_str());
    Serial.printf("✓ DNS: %s\n", WiFi.dnsIP().toString().c_str());
    Serial.printf("✓ Signal Strength: %d dBm\n", WiFi.RSSI());
    Serial.printf("✓ MAC Address: %s\n", WiFi.macAddress().c_str());
    Serial.println("----------------------\n");
}

int WiFiManager::getRSSI() {
    return WiFi.RSSI();
}