/*
 * ESP32 → IOT LAB Inference API
 * Reads sensors every 60s, POSTs to /predict/irrigation + /predict/anomaly,
 * drives relay (D5) based on response.
 *
 * Wiring (adjust pins to your board):
 *   Soil moisture AO -> GPIO34
 *   DHT22 DATA       -> GPIO4
 *   pH analog        -> GPIO35
 *   TDS analog       -> GPIO32
 *   NPK (RS485)      -> GPIO16/17 via MAX485
 *   Relay IN         -> GPIO5
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

#define WIFI_SSID       "YOUR_SSID"
#define WIFI_PASS       "YOUR_PASS"
#define API_BASE        "http://192.168.1.10:8000"
#define CROP_TYPE       "tomato"

#define PIN_SOIL   34
#define PIN_DHT    4
#define PIN_PH     35
#define PIN_TDS    32
#define PIN_RELAY  5

DHT dht(PIN_DHT, DHT22);

void setup() {
  Serial.begin(115200);
  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, LOW);
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi OK");
}

float readSoilMoisturePct() {
  int raw = analogRead(PIN_SOIL);          // 0..4095
  return map(raw, 4095, 0, 0, 100);        // dry..wet -> 0..100
}

float readPH()  { return analogRead(PIN_PH)  * 14.0 / 4095.0; }
float readTDS() { return analogRead(PIN_TDS) * 1000.0 / 4095.0; }

bool postPredict(const char* path, JsonDocument& body, JsonDocument& out) {
  HTTPClient http;
  http.begin(String(API_BASE) + path);
  http.addHeader("Content-Type", "application/json");
  String payload; serializeJson(body, payload);
  int code = http.POST(payload);
  if (code != 200) { Serial.printf("%s -> %d\n", path, code); http.end(); return false; }
  deserializeJson(out, http.getString());
  http.end();
  return true;
}

void loop() {
  float moist = readSoilMoisturePct();
  float temp  = dht.readTemperature();
  float hum   = dht.readHumidity();
  float ph    = readPH();
  float tds   = readTDS();

  StaticJsonDocument<256> body;
  body["soil_moisture"] = moist;
  body["temperature"]   = temp;
  body["humidity"]      = hum;
  body["ph"]            = ph;
  body["tds"]           = tds;
  body["rainfall"]      = 0;
  body["crop_type"]     = CROP_TYPE;

  StaticJsonDocument<256> irr;
  if (postPredict("/predict/irrigation", body, irr)) {
    bool pump = irr["irrigate"] | false;
    digitalWrite(PIN_RELAY, pump ? HIGH : LOW);
    Serial.printf("irrigate=%d conf=%.2f\n", pump, (float)irr["confidence"]);
  }

  StaticJsonDocument<256> anom;
  if (postPredict("/predict/anomaly", body, anom)) {
    if (anom["is_anomaly"] | false) Serial.println("!! sensor anomaly");
  }

  delay(60000);
}
