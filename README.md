// ═══════════════════════════════════════════
// HC-SR04 Ultrasonic Sensor + Buzzer
// Arduino UNO
// TRIG → D9  |  ECHO → D10  |  BUZZER → D8
// ═══════════════════════════════════════════

const int TRIG_PIN  = 9;
const int ECHO_PIN  = 10;
const int BUZZ_PIN  = 8;

// Distance threshold in cm — buzzer triggers below this
const int THRESHOLD = 30;

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZ_PIN, OUTPUT);
  Serial.begin(9600);
}

long getDistance() {
  // Send 10µs trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo — timeout 30ms
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  if (dur == 0) return -1;  // out of range
  return dur / 58;             // convert µs → cm
}

void loop() {
  long dist = getDistance();

  if (dist == -1) {
    // Out of range
    Serial.println("Out of range");
    noTone(BUZZ_PIN);

  } else if (dist < THRESHOLD) {
    // OBSTACLE — beep faster as object gets closer
    Serial.print("OBSTACLE: ");
    Serial.print(dist);
    Serial.println(" cm");

    // Beep frequency increases as distance decreases
    int freq = map(dist, 2, THRESHOLD, 2000, 500);
    tone(BUZZ_PIN, freq);
    delay(100);
    noTone(BUZZ_PIN);
    delay(map(dist, 2, THRESHOLD, 50, 300));

  } else {
    // Clear path
    Serial.print("Clear: ");
    Serial.print(dist);
    Serial.println(" cm");
    noTone(BUZZ_PIN);
    delay(200);
  }
}
