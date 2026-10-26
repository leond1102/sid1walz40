# Station – Kurzfassung

Stand: 25.10.2026 · Ausführlich: [DOKUMENTATION.md](DOKUMENTATION.md) · Eine Seite:
[UEBERSICHT_A4.pdf](UEBERSICHT_A4.pdf) · Anleitung: [INBETRIEBNAHME.md](INBETRIEBNAHME.md)

## Was die Station macht

Ein vierbeiniger Roboter (**SID**, DEEP Robotics Lite3) mit einem Greifer auf dem Rücken verteilt
drei Metallstützen auf feste Plätze und holt sie danach wieder ab. Gesteuert wird alles von
einem Linux-PC.

**Ein kompletter Ablauf (`start`):**

1. **Ausliefern** (dreimal): Die YOLO-Kamera erkennt eine Stütze auf der Erkennungsfläche.
   Ein Mensch stellt sie innerhalb von 10 s auf das Fließband, das Band fährt sie 18 s lang
   zur Dockingstation. Der SID fährt ins Dock, greift die Stütze, fährt rückwärts heraus und
   stellt sie auf ihren Platz: (2.0, −0.6), (2.0, 0.0) oder (2.0, 0.6).
2. **Pause:** Der SID wartet 30 s bei (1.5, 0).
3. **Zurückholen** (dreimal): Der SID fährt an eine Stütze, greift sie und bringt sie zur
   Abgabe (1.0, 0.25) auf der Erkennungsfläche.

`liefern` führt nur Schritt 1 aus, `holen` nur Schritt 3. Der Ablauf ist eine feste Liste
von 68 Schritten; das Frontend zeigt jeweils den vorherigen, aktuellen und nächsten.

## Komponenten

| Komponente | Anbindung | Modul |
|---|---|---|
| SID (Lite3) | UDP, `192.168.1.120:43893` | `scripts/navigate.py`, `scripts/lite3_commander.py` |
| SID-Ortung (Arducam an der Decke, ArUco-Marker) | USB → MQTT | `scripts/aruco_tracker_calibrated.py` |
| Greifer (Linearachse + Servozange, ESP32-C6) | MQTT | `scripts/arm_controller.py`, `scripts/arm_config.py` |
| Fließband (ESP32-C6) | MQTT | `scripts/conveyor_controller.py` |
| Stützenerkennung (IR-Kamera + YOLO) | USB | `scripts/F_stilt_erkennung_YOLO.py` |
| MQTT-Broker | `192.168.2.1:1883` | – |
| Bedienoberflächen | Browser, `localhost:8080` / `:8081` | `scripts/station_frontend.py`, `scripts/trainer_server.py` |

## Wie der SID navigiert

Die Kamera an der Decke misst die Position des SID anhand von ArUco-Markern. Aus gemessenen
Probeläufen wurden Laufmodelle gelernt (`models/dynamics_model.joblib`,
`models/rotation_model.joblib`). Mit diesen Modellen plant der Rechner jeden Schritt einzeln:
messen, einen Bewegungsbefehl ausführen, anhalten, neu messen.

**Sicherheitszone:** Alles mit x < 1,2 m ist gesperrt. Nur das Ein- und Ausdocken und die
Abgabe führen dort hinein, und zwar jeweils in einer geraden Linie ohne Drehen. Fällt die
Ortung aus (keine Pose seit 1,5 s), bleibt der SID sofort stehen.

## Station bedienen

Doppelklick auf das Symbol **Station** auf dem Schreibtisch. Es öffnen sich ein Terminal
(Meldungen, Befehle) und das **Frontend** im Vollbild. Läuft die Station schon, öffnet das
Symbol nur das Frontend. Autostart beim Hochfahren: `./station_desktop.sh autostart-on`
(aus: `autostart-off`). Von Hand: `python main.py` (`--sim-nav` = SID simuliert).

| Knopf im Frontend | Taste | Terminal / MQTT | Wirkung |
|---|---|---|---|
| START | F2 | `start` / `liefern` / `holen` | Ablauf starten (nur im Zustand `idle`) |
| PAUSE / WEITER | F3 | `pause` / `resume` | nach dem laufenden Schritt anhalten / weiter |
| STOP | F4 | `stop` | laufende Aktion sauber abbrechen |
| QUITTIEREN | F8 | `reset` | Fehlermeldungen quittieren, Fehler / Not-Aus zurücksetzen |
| NOT-AUS | Leertaste | `estop` | alles sofort stoppen |
| BEENDEN | F10 | `quit` | Programm beenden, alle Geräte stoppen |

Die Tasten sind über ⚙ im Frontend oder in `config/hotkeys.yaml` einstellbar. Weitere
Terminal-Befehle: `undock` (SID aus Dock oder Zone fahren), `init` (Startpose neu messen),
`status`. MQTT-Befehle gehen auf `station/cmd`, den Zustand meldet die Station als JSON auf
`station/state`.

**Frontend:** zwei Kamerabilder (Greifarm/YOLO und Ortung), Ablauf mit Fortschritt,
Zustandskette, Fehlermeldungen und rechts oben **Systeme**: Broker, Ortung, SID, Band,
Greifer und YOLO in grün/gelb/rot mit Grund. Das Frontend startet als Erstes und läuft auch,
wenn Broker oder Geräte fehlen; diese verbinden sich von selbst nach, und die Startpose wird
automatisch gemessen, sobald die Ortung den SID sieht.

## Neue Bewegungen trainieren

Symbol **Bewegungstraining** (oder `python trainer.py`, `http://localhost:8081/`), nur wenn die
Station **nicht** läuft.

1. **Sammeln:** Policy wählen (Geradlinig, Gemischt, Drehungen, Feine Schritte), Anzahl
   Samples, starten. Der SID läuft Zufallsbefehle im Umkreis von 1 m und kehrt selbst zum
   Start zurück. Ortungsbild, Fortschritt und Abstand zum Start sind live zu sehen.
   **STOPP = Leertaste.** Simulation geht ohne Roboter.
2. **Trainieren:** Lauf- oder Drehmodell, Datensätze ankreuzen, trainieren. Ergebnis ist ein
   Kandidat in `models/kandidaten/`.
3. **Übernehmen:** sichert das alte Modell in `models/archiv/`; das neue gilt ab dem nächsten
   Start der Station.

## Ordner

| Ordner | Inhalt |
|---|---|
| `models/` | aktive Modelle (Lauf, Drehung, YOLO), `kandidaten/`, `archiv/` |
| `data/` | `datasets/` (Trainingsdaten), `nav_logs/` (Fahrlogs), `plots/`, `archiv/` |
| `scripts/` | alle Module, `frontend/` (Oberflächen) |
| `config/` | `frontend.yaml`, `hotkeys.yaml` |
| `logs/` | eine Logdatei pro Programmstart – bei Fehlern zuerst hier nachsehen |
