"""Zentrale Konfiguration des Greifarms (ESP32-C6).

Alle Zahlenwerte und Grenzen stehen hier. Der ESP32 kennt keine Grenzen mehr:
arm_controller.py prüft jeden Befehl gegen diese Werte und schickt die
Bewegungsparameter beim Verbinden als retained Nachricht an den ESP32.

Nach einer Änderung in der Konsole 'config' eingeben, dann werden die Werte
neu geladen und sofort an den ESP32 geschickt (nur im Stillstand).
Änderungen an BROKER, PORT und TOPIC_BASE brauchen einen Neustart des Skripts.
"""

# --- MQTT ---
BROKER = "192.168.2.1"
PORT = 1883
TOPIC_BASE = "station/arm"

# Heartbeat kommt alle 3 s, nach 10 s gilt der ESP32 als nicht erreichbar
HEARTBEAT_TIMEOUT_S = 10.0

# Befehle mit wait=True warten höchstens so lange auf die Fertigmeldung
# (volle Referenzfahrt: 20000 Schritte * 400 us = 8 s)
WAIT_TIMEOUT_S = 20.0

# --- Achse: Mechanik ---
STEPS_PER_REV = 200
CM_PER_REV = 0.8

# --- Achse: Grenzen ---
# 0 Schritte = Limit-Schalter. ACHTUNG: am oberen Ende gibt es KEINEN Schalter,
# AXIS_MAX_STEPS ist dort die einzige Absicherung.
AXIS_MIN_STEPS = 0
AXIS_MAX_STEPS = 11000

# --- Achse: Referenzfahrt und Geschwindigkeit (gehen an den ESP32) ---
HOMING_MAX_STEPS = 20000   # Sicherheitsabbruch der Referenzfahrt
BACKOFF_STEPS = 150        # Freifahren vom Schalter

# Halbe Schrittperiode: 200 us HIGH + 200 us LOW = 400 us pro Schritt -> 2500 Schritte/s
STEP_HALF_US = 200

# --- Greifer: Grenzen ---
GRIP_CLOSED_US = 980       # Minimum (zu)
GRIP_OPEN_US = 1600        # Maximum (offen)

# --- Greifer: Servo (gehen an den ESP32) ---
SERVO_MIN_US = 500         # Pulsbereich für Servo.attach()
SERVO_MAX_US = 2500
SERVO_TRAVEL_MS = 800      # Zeit für eine Servobewegung

# Beim Start den Greifer öffnen. Auf False setzen, falls beim Booten ein
# Teil im Greifer liegen kann, das nicht fallen darf.
OPEN_GRIPPER_ON_BOOT = True

# --- Teileerkennung (induktiver Sensor, gehen an den ESP32) ---
# False = Sensor wird ignoriert (defekt): Greifen gilt ohne Prüfung als erfolgreich,
# Meldungen wie "part_lost" lösen keinen Fehler aus.
USE_PART_SENSOR = False
SENSOR_LOSS_MS = 500           # Signalausfall bis zur Meldung "part_lost"
GRIP_SETTLE_MS = 300           # Beruhigung nach dem Schließen
GRIP_SAMPLES = 5               # Messungen für die Griffprüfung
GRIP_SAMPLE_INTERVAL_MS = 10
