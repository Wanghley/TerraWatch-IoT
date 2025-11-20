#include "wifi_ap_manager.h"

WiFiAPManager::WiFiAPManager() {}

void WiFiAPManager::begin(const char* ssid, const char* password, uint8_t channel, bool hidden, uint8_t maxConnections) {
    Serial.println("\n=== Starting WiFi Access Point ===");
    Serial.printf("SSID: %s\n", ssid);
    Serial.printf("Password: %s\n", (strlen(password) > 0 ? password : "(open)"));
    Serial.printf("Channel: %d | Hidden: %d | Max Conn: %d\n", channel, hidden, maxConnections);

    // Turn off STA if active
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    delay(500);

    // Start AP mode
    WiFi.mode(WIFI_AP);
    // For ESP32, softAP has signature: softAP(ssid, password, channel, hidden, max_connection)
    bool ok = WiFi.softAP(ssid, (strlen(password) > 0 ? password : NULL), channel, hidden, maxConnections);
    if (!ok) {
        Serial.println("! softAP() returned false");
    }

    // Wait for setup
    delay(1000);

    printAPInfo();
}

void WiFiAPManager::printAPInfo() {
    Serial.println("\n--- Access Point Info ---");
    Serial.printf("✓ SSID: %s\n", WiFi.softAPSSID().c_str());
    Serial.printf("✓ IP Address: %s\n", WiFi.softAPIP().toString().c_str());
    Serial.printf("✓ MAC Address: %s\n", WiFi.softAPmacAddress().c_str());
    Serial.println("-------------------------\n");
}

bool WiFiAPManager::isAPRunning() {
    return WiFi.getMode() == WIFI_MODE_AP || WiFi.getMode() == WIFI_MODE_APSTA;
}

bool WiFiAPManager::sendJSON(const char* serverUrl, const String& jsonPayload, uint16_t timeoutMs) {
    if (!isAPRunning()) {
        Serial.println("sendJSON: AP not running. Call begin() first.");
        return false;
    }

    // HTTPClient is used to POST JSON
    HTTPClient http;
    WiFiClient client;

    Serial.printf("Sending JSON to: %s\n", serverUrl);
    Serial.printf("Payload: %s\n", jsonPayload.c_str());

    // begin accepts (WiFiClient&, url)
    if (!http.begin(client, serverUrl)) {
        Serial.println("HTTPClient.begin() failed");
        return false;
    }

    http.setConnectTimeout(timeoutMs);
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(jsonPayload);
    if (httpCode <= 0) {
        Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
        http.end();
        return false;
    }

    Serial.printf("HTTP response code: %d\n", httpCode);
    if (httpCode >= 200 && httpCode < 300) {
        String response = http.getString();
        Serial.printf("Server response: %s\n", response.c_str());
        http.end();
        return true;
    } else {
        String resp = http.getString();
        Serial.printf("Server returned non-2xx: %d -> %s\n", httpCode, resp.c_str());
        http.end();
        return false;
    }
}

void WiFiAPManager::disconnect() {
    if (isAPRunning()) {
        Serial.println("Stopping WiFi AP...");
        WiFi.softAPdisconnect(true);
        WiFi.mode(WIFI_OFF);
        delay(500);
    } else {
        Serial.println("disconnect(): AP not running.");
    }
}
