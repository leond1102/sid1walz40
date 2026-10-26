# Station – Technische Dokumentation

Stand: 25.10.2026 · Einstiegspunkte: `main.py` (Station), `trainer.py` (Bewegungstraining)

Diese Dokumentation beschreibt Aufbau, Ablauf, Zustandsautomat, Schnittstellen, Bedienoberflächen
und Parameter der Station. Weitere Fassungen:

- [KURZFASSUNG.md](KURZFASSUNG.md): das Wichtigste auf zwei Seiten
- [UEBERSICHT_A4.pdf](UEBERSICHT_A4.pdf): Funktion des Codes auf einer A4-Seite (Quelle: `UEBERSICHT_A4.html`)
- [INBETRIEBNAHME.md](INBETRIEBNAHME.md): Schritt-für-Schritt-Anleitung und Troubleshooting

---

## Inhalt

1. [Aufgabe der Station](#1-aufgabe-der-station)
2. [Hardware und Netzwerk](#2-hardware-und-netzwerk)
3. [Softwareaufbau](#3-softwareaufbau)
4. [Koordinatensystem und Arbeitsbereich](#4-koordinatensystem-und-arbeitsbereich)
5. [Der Ablauf im Detail](#5-der-ablauf-im-detail)
6. [Zustandsautomat](#6-zustandsautomat)
7. [Befehle](#7-befehle)
8. [MQTT-Schnittstelle](#8-mqtt-schnittstelle)
9. [Die Geräte im Detail](#9-die-geräte-im-detail)
10. [Anzeige und Logging](#10-anzeige-und-logging)
11. [Konfiguration: wo steht was?](#11-konfiguration-wo-steht-was)
12. [Laufmodelle: Daten sammeln und trainieren](#12-laufmodelle-daten-sammeln-und-trainieren)
13. [Test- und Hilfswerkzeuge](#13-test--und-hilfswerkzeuge)
14. [Bekannte Einschränkungen](#14-bekannte-einschränkungen)

---

## 1. Aufgabe der Station

Die Station verteilt drei Metallstützen mit einem vierbeinigen Roboter (**SID**, DEEP Robotics
Lite3) auf feste Plätze und sammelt sie danach wieder ein. Auf dem SID sitzt ein Greifer
aus einer senkrechten Linearachse und einer Servozange. Ein Fließband bringt die Stützen zur
Dockingstation. Dort übernimmt sie der SID. Eine IR-Kamera mit YOLO-Modell erkennt, ob eine
Stütze auf der Erkennungsfläche bereitsteht.

Den Roboterarm, der die Stützen auf das Band legen soll, gibt es noch nicht. Diesen Schritt
macht deshalb vorerst ein Mensch: Die Station fordert ihn dazu auf und wartet 10 s.

Zur Software gehören zwei Programme mit eigener Bedienoberfläche im Browser (WALZ-4.0-Design):

| Programm | Aufgabe | Oberfläche |
|---|---|---|
| `main.py` – **Station** | Ablauf steuern, Geräte überwachen, Fehler melden | `http://localhost:8080/` (Abschnitt 10.1) |
| `trainer.py` – **Bewegungstraining** | Fahrdaten des SID sammeln, Laufmodelle trainieren und übernehmen | `http://localhost:8081/` (Abschnitt 12.1) |

Beide starten per Desktop-Symbol und dürfen **nicht gleichzeitig** den SID steuern.

---

## 2. Hardware und Netzwerk

### 2.1 Geräte

| Gerät | Aufgabe | Anbindung | Adresse / Kennung |
|---|---|---|---|
| **SID** (DEEP Robotics Lite3) | läuft, dockt, trägt den Greifer | UDP (eigenes Protokoll) | `192.168.1.120`, Port `43893`; lokal Port `12345` |
| **Arducam** (Decke) | Ortung des SID über ArUco-Marker | USB, MJPG 1920×1080 | Gerätename enthält `arducam` |
| **IR-Kamera** | YOLO-Erkennung der Stützen | USB, YUYV 340×340 | erster `/dev/videoN`, der YUYV-Bilder liefert |
| **Greifer** (ESP32-C6) | Linearachse (Schrittmotor + Endschalter unten) und Servozange | WLAN → MQTT | Topics `station/arm/...` |
| **Fließband** (ESP32-C6) | fährt eine Stütze zur Dockingstation | WLAN → MQTT | Topics `station/conveyor...` |
| **MQTT-Broker** | Nachrichtenzentrale aller Geräte | TCP | `192.168.2.1:1883` |
| **Steuer-PC** | führt `main.py` bzw. `trainer.py`, Ortung, YOLO und die Frontends aus (Ubuntu 22.04, Firefox) | – | muss `192.168.1.x` **und** `192.168.2.x` erreichen |

YOLO läuft auf der CPU. Die eingebaute Grafikkarte (Quadro K620, sm_50) wird von aktuellen
PyTorch-Versionen nicht mehr unterstützt und ist deshalb im Code ausgeblendet
(`CUDA_VISIBLE_DEVICES=""`).

### 2.2 ArUco-Marker

| ID | Lage | Bedeutung |
|---|---|---|
| 0 | auf dem SID | Robotermarker (Position und Blickrichtung) |
| 1 | am Boden | Ursprung des Koordinatensystems |
| 2 | am Boden | liegt auf der +X-Achse |
| 3 | am Boden | liegt auf der −Y-Achse |

Wörterbuch `DICT_4X4_50`, Kantenlänge des schwarzen Quadrats 10 cm. Die Markerebene auf dem
SID liegt fest 1,70 m unter der Kamera (`ROBOT_MARKER_CAMERA_DISTANCE_M`).

---

## 3. Softwareaufbau

### 3.1 Dateien

```
main.py                         Zentrale Steuerung: Zustandsautomat, Ablauf, Befehle, MQTT
trainer.py                      Bewegungstraining mit eigenem Frontend (siehe 12)
start_station.sh, start_trainer.sh   Start über die Desktop-Symbole
station_desktop.sh              Desktop-Symbole und Autostart einrichten
logs/                           station_*.log / trainer_*.log pro Programmstart (alles, DEBUG)

models/                         aktive Modelle (Pfade zentral in scripts/paths.py)
  dynamics_model.joblib         Laufmodell (alle Lauf-Primitive)
  rotation_model.joblib         Drehmodell (präzise Drehungen)
  best.pt                       YOLO-Modell der Stützenerkennung
  kandidaten/                   frisch trainierte Modelle, noch nicht übernommen
  archiv/                       ersetzte Modelle
data/
  datasets/                     gesammelte Trainingsdaten: *_dataset.csv, *_return_log.csv (+ *_sim.csv)
    meta/                       Einstellungen jedes Sammellaufs (*_meta_<zeit>.json)
  nav_logs/                     nav_log.csv, nav_rotation_log.csv: jede ausgeführte Bewegung (zum Nachtrainieren)
  plots/                        Diagramme aus dem Training
  archiv/                       alte Daten (falsch kalibriert), nicht mehr benutzen

config/
  frontend.yaml                 Frontend: Adresse, Browser, Quellen der Kamerabilder
  hotkeys.yaml                  Tastenkürzel des Frontends (auch im Frontend über ⚙ änderbar)
assets/                         Symbole für den Schreibtisch
docs/                           Dokumentation, Logo, bilder/

scripts/
  paths.py                      wo Modelle und Daten liegen (unabhängig vom Arbeitsordner)
  navigate.py                   Navigation des SID (Planer, Sicherheitszone, Docken)
  lite3_commander.py            UDP-Befehle an den SID (Heartbeat, Achsen X/Y/Yaw)
  aruco_tracker_calibrated.py   SID-Ortung: Kamera -> Pose per MQTT (mit --preview: Bild fürs Frontend)
  arm_controller.py             Greifer: Grenzen prüfen, Befehle senden, Zustand verfolgen
  arm_config.py                 alle Zahlenwerte des Greifers
  conveyor_controller.py        Fließband: Fahrbefehle und Status
  F_stilt_erkennung_YOLO.py     IR-Kamera + YOLO-Modell (StiltDetector)
  station_display.py            farbige Konsole, Logdatei, altes Kamerafenster, Start der Ortung
  station_frontend.py           Frontend der Station: Webserver (Kamerabilder, Ablauf, Systeme, Fehler, Knöpfe)
  trainer_server.py             Frontend des Bewegungstrainings: Webserver + Steuerung der Sammel-/Trainingsläufe
  frontend/                     Oberflächen im Browser (WALZ-4.0-Design): index.html/app.* (Station),
                                trainer.* (Bewegungstraining); walz_logo.png = Kopie von docs/walz_4.0_logo-horizontal.png
  station_statemachine.py       älteres Grundgerüst des Automaten (nur Navigation, Beispiel)
  dynamics_model.py / rotation_model.py      Modellklassen (für joblib nötig)
  collect_*.py, train_*.py      Datensammlung und Training der Laufmodelle
  arm_console.py, conveyor_console.py        Handsteuerung einzelner Geräte
  check_heading.py              Prüft die Winkelmessung der Ortung
  arducam/                      Kalibrierbilder und -skripte der Arducam
```

Die Gerätemodule liegen in `scripts/` und importieren sich dort gegenseitig. `main.py`
hängt dieses Verzeichnis an den Suchpfad (`sys.path`) an. Alle Modell- und Datenpfade kommen
aus `scripts/paths.py` und hängen nicht vom Arbeitsordner ab; bei den Skripten reicht der
Dateiname (`--csv linear_dataset.csv` wird in `data/datasets/` und `data/nav_logs/` gesucht).

### 3.2 Zusammenspiel

```
                        ┌───────────────── Steuer-PC ──────────────────┐
 Tastatur ──────────────┤ main.py                                      │
 Frontend (Browser) ◄───┤  StationFrontend :8080 (station_frontend.py) │
 (Leertaste = NOT-AUS) ─┤  Station (LockedMachine, transitions)        │
                        │   ├─ NavigationController (navigate.py) ─────┼── UDP ──► SID (Lite3)
                        │   │    └─ PoseTracker  ◄── MQTT pose ──┐     │
                        │   ├─ ArmController ◄──── MQTT ─────────┼─────┼──► Greifer-ESP32
                        │   ├─ ConveyorRemote ◄─── MQTT ─────────┼─────┼──► Fließband-ESP32
                        │   └─ StiltDetector ◄── USB ── IR-Kamera│     │
                        │                                        │     │
                        │ aruco_tracker_calibrated.py (Prozess) ─┘     │
                        │   ◄── USB ── Arducam   ──► pose, status, preview │
                        └──────────────────────────────────────────────┘
                                      ▲  station/cmd  │ station/state
                                      └── MQTT-Broker 192.168.2.1 ──►
```

`trainer.py` (Bewegungstraining) ist ein eigenes Programm: Es startet die Skripte
`collect_*.py` / `train_*.py` als Unterprozesse und zeigt das Ortungsbild per MQTT
(Abschnitt 12.1). Es benutzt weder den Zustandsautomaten noch die Geräte der Station.

### 3.3 Threads

| Thread | Aufgabe |
|---|---|
| Hauptthread | wartet auf das Programmende (mit `--cv-window`: Kamerafenster, OpenCV verlangt den Hauptthread) |
| `frontend` + je Browserverbindung einer | Webserver der Bedienoberfläche (Zustand, MJPEG-Bilder, Befehle) |
| `feed-*` | liest die Kamerabilder fürs Frontend (nur solange jemand zuschaut) |
| `systeme` | prüft jede halbe Sekunde alle Teilsysteme für die Systemübersicht, meldet Ausfälle |
| `start` | wartet beim Programmstart auf die Ortung, misst die Startpose, misst bei Bedarf neu |
| `tastatur` | liest Befehle von `stdin` (mit Timeout, damit das Beenden nicht hängt) |
| `ablauf` | arbeitet die Schrittliste von `start` / `liefern` / `holen` ab |
| `band` (ThreadPool, 1 Worker) | sucht die nächste Stütze und fährt das Band, **parallel** zum SID |
| `nav-*` / `task-*` | Einzelbefehle (Navigation bzw. Geräte) |
| MQTT-Netzwerkschleifen | je Client (Station, Greifer, Band, Vision, Pose, Frontend) |

Der Automat ist eine `LockedMachine`, Übergänge aus verschiedenen Threads sind also
threadsicher. Arbeits-Threads melden ihr Ergebnis über einen Trigger zurück
(`cycle_done`, `nav_failed`, `task_done` …). Passt ein Trigger nicht zum aktuellen Zustand
(z. B. weil inzwischen ein Not-Aus kam), ignoriert `_safe_trigger()` ihn.

### 3.4 Programmstart (`main()`)

1. Logging einrichten (`logs/station_JJJJMMTT_HHMMSS.log`).
2. **Frontend starten** und im Vollbild öffnen (siehe 10.1) – als Erstes, damit der Start
   zu sehen ist. Das Frontend läuft ab hier bis zum Programmende, egal was danach schiefgeht.
3. Laufmodelle laden, SID anlegen. Klappt das nicht (z. B. Modelldatei fehlt, UDP-Port
   belegt), läuft das Programm ohne Station weiter: Frontend zeigt `FAILED` und den Grund in
   der Systemübersicht, Befehle werden abgelehnt. Ursache beheben und neu starten.
4. Mit dem MQTT-Broker verbinden (Station, Fließband, Greifer, Ortungs-Posen, YOLO) – **im
   Hintergrund**: Ist der Broker nicht erreichbar, startet trotzdem alles und verbindet sich
   von selbst, sobald er da ist.
5. Nur im echten Betrieb: SID-Ortung starten (`aruco_tracker_calibrated.py` als eigener
   Prozess), falls sie noch nicht läuft. Mit Frontend ohne eigenes Fenster, das Debugbild
   kommt per MQTT (`--headless --preview`) ins Frontend; mit `--cv-window` wie früher mit Fenster.
6. YOLO-Modell laden und IR-Kamera öffnen. Fehlt eines davon, läuft das Programm trotzdem
   weiter, aber `start` und `liefern` werden abgelehnt.
7. Tastatur-Thread starten (mit `--cv-window`: Kamerafenster öffnen). Parallel im Thread
   `start`: bis zu 60 s warten, bis die Ortung `tracking` meldet, dann `initialize()`
   (Startpose messen, erster Zustand `idle`, `docked` oder `error`).
8. Fehlte die Startpose (Broker oder Ortung nicht da), steht die Station in `error`. Sie
   misst **automatisch neu**, sobald die Ortung `tracking` meldet (frühestens alle
   `INIT_RETRY_S` = 15 s), oder sofort mit `init` / `reset` / QUITTIEREN.
9. Nach 8 s meldet die Systemübersicht einmal, was nicht bereit ist; danach jeden Ausfall
   und jede Wiederkehr eines Systems.

Eine Sicherheitsabfrage („Tippe START“) gibt es nicht mehr. Der SID bewegt sich erst, wenn
ein Bewegungsbefehl kommt (`start`, `liefern`, `holen`, `goto`, `dock` …).

**Beenden:** BEENDEN-Knopf (F10) im Frontend, `quit` im Terminal, Strg+C, Schließen des
Terminalfensters (SIGHUP) oder `kill` (SIGTERM). In allen Fällen werden laufende Aktionen
abgebrochen, Band und Greifer gestoppt, der SID angehalten, `station/state` auf `offline`
gesetzt und eine von `main.py` gestartete Ortung sowie das Frontend-Fenster beendet. Ohne
Broker dauert das etwa 5 s (Verbindungs-Timeout). Wird nur das Frontend-Fenster geschlossen,
läuft die Station weiter (Adresse steht im Terminal).

### 3.5 Start über den Schreibtisch und Autostart

| Datei | Aufgabe |
|---|---|
| `start_station.sh` | startet `main.py` (Argumente werden durchgereicht). Läuft die Station schon, öffnet es nur das Frontend. Endet `main.py` mit Fehler, bleibt das Terminal offen. |
| `start_trainer.sh` | dasselbe für `trainer.py` |
| `station_desktop.sh icon` | legt die Symbole **Station** und **Bewegungstraining** auf den Schreibtisch (startfähig markiert) |
| `station_desktop.sh autostart-on` / `autostart-off` / `status` | Station nach dem Anmelden automatisch starten (`~/.config/autostart/station.desktop`, 10 s Verzögerung) |

Die Symbole öffnen ein Terminalfenster (Meldungen, Befehle) und das Frontend im Vollbild.
Auf dem Steuer-PC ist die automatische Anmeldung aktiv; Autostart bedeutet dort also Start
beim Hochfahren. Symbole: `assets/station.svg`, `assets/trainer.svg`.

---

## 4. Koordinatensystem und Arbeitsbereich

Alle Positionen sind **absolute Kamerakoordinaten** in Metern bzw. Grad, gemessen vom
Ursprungsmarker ID 1. Gemeint ist immer die Position des Robotermarkers auf dem SID.
θ = 0° zeigt in +X-Richtung (weg von der Station), positive Winkel drehen Richtung +Y.

```
  y
  ▲                 Sicherheitszone │  freier Bereich
  │                   (x < 1,2 m)   │
  │   0.6 ┤                         │              ● Platz 3 (2.0, 0.6)
  │  0.25 ┤  Abgabe ◄━━━━━━━━━━━━━━━┥ (1.0, 0.25)
  │   0.0 ┤                         │     ◆ Pause (1.5, 0)   ● Platz 2 (2.0, 0.0)
  │ −0.25 ┤  Dock ◄━━━━━━━━━━━━━━━━━┥ Vorposition (1.2, −0.25, 180°)
  │ −0.6  ┤  (0.15, −0.25)          │              ● Platz 1 (2.0, −0.6)
  └───────┴─────────────────────────┴──────────────────────────────► x
          0                        1.2           1.5       2.0
```

Die Skizze ist nicht maßstäblich. Wo +Y im Raum liegt, legt Marker 3 fest (er liegt auf −Y).

| Position | Koordinaten | Parameter (Datei) |
|---|---|---|
| Plätze der Stützen | (2.0, −0.6), (2.0, 0.0), (2.0, 0.6), Blick 0° | `PLACE_POSITIONS`, `PLACE_ANGLE` (`main.py`) |
| Vorposition vor einem Platz | 0,5 m vor dem Platz | `PLACE_STANDOFF` (`main.py`) |
| Warteplatz (Pause) | (1.5, 0.0) | `PAUSE_POSITION` (`main.py`) |
| Abgabe (Erkennungsfläche) | (1.0, 0.25), Einfahrt mit Blick 180° | `DROP_POSITION` (`main.py`) |
| Dock-Vorposition | (1.20, −0.25, 180°) | `DOCK_PRE_POSE` (`navigate.py`) |
| Dock | (0.15, −0.25) | `DOCK_POSE` (`navigate.py`) |
| Sicherheitszone | x < 1,2 m | `SAFETY_ZONE_X` (`navigate.py`) |
| Sicherheitsradius | 4,0 m um die Startpose | `MAX_DISTANCE_FROM_START` (`navigate.py`) |

### Regeln der Sicherheitszone

- Ziele in der Zone lehnt `go_to` / `approach` ab (`REJECTED`).
- Der Planer wählt keine Bewegung, die laut Modell in der Zone enden würde.
- Während jeder Bewegung wird die Live-Pose geprüft: Gerät der SID in die Zone
  (mit 5 cm Marge), hält er sofort an (`SAFETY_STOP`).
- Nur `dock`, `enter_zone` (Abgabe) und `undock` führen in die Zone. Sie laufen immer in
  **einer geraden Linie ohne Drehung**. Die Drehsperre bleibt aktiv, bis `undock` den SID
  wieder auf x ≥ 1,3 m gebracht hat.
- Steht der SID beim Programmstart in der Zone, ist nur `undock` erlaubt.

---

## 5. Der Ablauf im Detail

`start` (Modus `alles`) = Ausliefern → Pause → Zurückholen.
`liefern` = nur Ausliefern, `holen` = nur Zurückholen.
Auch `start liefern` und `start holen` funktionieren.

**Voraussetzungen:** Die Station ist im Zustand `idle`, der SID steht also frei außerhalb
der Zone. Der Greifer muss erreichbar sein, für `start` und `liefern` zusätzlich Fließband
und YOLO-Kamera. Fehlt etwas, wird der Befehl mit Begründung abgelehnt.

### 5.1 Ausliefern (für jede der 3 Stützen)

Zu Beginn einmal: Greifer-Referenzfahrt und Achse ganz nach oben.

Zwei Stränge laufen **parallel**:

| Band-Strang (Thread `band`) | SID-Strang (Thread `ablauf`) |
|---|---|
| 1. Kamera sucht alle 0,5 s, bis YOLO mindestens eine Stütze auf der Erkennungsfläche sieht | 1. SID läuft zur Dock-Vorposition |
| 2. Aufforderung: „Stütze von Hand aufs Band stellen“, 10 s warten | 2. Greifer öffnet |
| 3. Band fährt 18 s → Stütze liegt am Bandende | 3. wartet, bis der Band-Strang fertig ist |
| | 4. SID fährt ins Dock, Greifer schließt, SID fährt rückwärts heraus |
| *(ab hier startet schon die Suche nach der nächsten Stütze)* | 5. SID trägt die Stütze zum Platz (Anfahren aus Richtung 0°, Vorposition 0,5 m davor) |
| | 6. Achse runter, Greifer auf, 0,3 m rückwärts, Greifer zu, Achse hoch |

Fehler des Band-Strangs (z. B. kein Kamerabild, Band meldet kein Fahrtende) werden beim
Warten in Schritt 3 an den Ablauf weitergereicht.

### 5.2 Pause

Der SID läuft zum Warteplatz (1.5, 0) und wartet 30 s.

### 5.3 Zurückholen (für jede der 3 Stützen, gleiche Reihenfolge)

1. Referenzfahrt des Greifers, falls er noch nicht referenziert ist (z. B. bei `holen`).
2. Zur Vorposition 0,5 m vor der Stütze.
3. Achse runter, Greifer auf.
4. Gerade vorwärts an die Stütze (nur Vorwärts-Primitive, höchstens 2 Schritte).
5. Greifer zu, Achse hoch.
6. Gerade in die Zone zur Abgabe (1.0, 0.25): Vorposition auf der Zonengrenze mit Blick 180°,
   dann ein gerader Vorwärtslauf.
7. Greifer auf, gerade rückwärts wieder heraus (`undock`).

### 5.4 Wartezeiten

| Parameter | Wert | Zweck |
|---|---|---|
| `GRIPPER_WAIT_S` | 2,0 s | nach jedem Öffnen/Schließen, bis die Zange sicher steht |
| `STEPPER_WAIT_S` | 1,0 s | nach jeder Fahrt der Achse (home/min/max) |
| `HAND_PLACE_S` | 10 s | Zeit, um die Stütze von Hand aufs Band zu stellen |
| `CONVEYOR_RUN_S` | 18 s | Fahrzeit des Bands bis zum Ende |
| `DETECT_RETRY_S` | 0,5 s | Pause zwischen zwei Suchversuchen der Kamera |
| `PAUSE_S` | 30 s | Pause zwischen Ausliefern und Zurückholen |
| `TRACKER_WAIT_S` | 60 s | Wartezeit auf die Ortung beim Programmstart |

Alle Wartezeiten sind unterbrechbar: `stop` oder `estop` beendet sie sofort.

### 5.5 Genauigkeit: Warnung oder Fehler?

Das Anfahren der Plätze, des Warteplatzes und der Vorpositionen beim Zurückholen läuft mit
`exact=False`. Ein ungenaues Ergebnis (`FAILED`) ist dort nur eine Warnung, der Ablauf geht
weiter. Docken, Ausdocken, Abgabe und Zurücksetzen müssen dagegen gelingen, sonst geht die
Station in `error`.

---

## 6. Zustandsautomat

### 6.1 Zustände

| Zustand | Bedeutung |
|---|---|
| `init` | Programmstart bzw. erneutes Messen der Startpose |
| `idle` | SID frei außerhalb der Zone, wartet auf Befehle |
| `running` | Ablauf läuft (Fortschritt in `phase` / `step` / `band`, alle Schritte in `plan`) |
| `paused` | Pause: der laufende Schritt wird noch fertig, dann hält der Ablauf an (`resume` = weiter) |
| `navigating` / `docking` / `undocking` | Einzelbefehl Navigation läuft |
| `conveying` / `detecting` / `arm_moving` / `gripping` | Einzelbefehl Gerät läuft |
| `docked` | SID steht im Dock |
| `error` | Fehler. Steht der SID in der Zone: erst `undock`, sonst `reset`. Fehlt die Startpose, wird sie automatisch neu gemessen, sobald die Ortung den SID verfolgt |
| `estop` | Not-Aus, nur `reset` |

### 6.2 Übergänge

```
               ┌──────── _started (frei) ──────────► idle ◄────────────────────┐
  init ────────┼──────── _started (im Dock) ───────► docked                    │
               └──────── _started (in Zone) / Fehler ► error                   │
                                                                               │
  idle ── start ──► running ── cycle_done / cycle_aborted ──► idle | docked | error
                    │      └── cycle_failed ─────────────────► error
                    └ pause ⇄ resume ┘  paused (Ergebnisse wie running, stop bricht ab)
  idle ── goto / approach ──► navigating ── Ergebnis ──► idle | error
  idle ── dock ─────────────► docking ───── Ergebnis ──► docked | idle | error
  docked | error (in Zone) ── undock ──► undocking ── Ergebnis ──► idle | docked | error
  idle | docked ── convey/detect/home/arm_to/grip/release ──► <Gerätezustand> ──► idle | docked | error
  Busy-Zustände ── stop ──► (bleibt, Ergebnis kommt als *_aborted)
  idle | docked | running | paused | Geräte ── fault (Greifer offline / Teil verloren) ──► error
  *  ── estop ──► estop
  error | estop ── reset ──► docked (im Dock) | idle (frei)
  error (ohne Startpose) ── reinit ──► init ── initialize() ──► idle | docked | error
```

Wichtige Einzelheiten:

- `nav_rejected` (Befehl unzulässig, nichts bewegt) führt zurück nach `idle`.
- **Startpose fehlt** (Broker oder Ortung beim Start nicht da): Zustand `error`. Der Thread
  `start` löst `reinit` aus, sobald die Ortung `tracking` meldet (frühestens alle 15 s,
  `INIT_RETRY_S`). Von Hand: `init`; auch `reset` und QUITTIEREN messen in diesem Fall neu,
  statt die Station ohne Startpose freizugeben.
- `nav_aborted` beim Docken/Ausdocken führt nach `error`, weil der SID dann in der
  Andockbahn stehen kann.
- `reset` funktioniert nur, wenn der SID **im Dock** oder **frei** steht. Steht er in der
  Zone, aber nicht im Dock, kommt die Meldung „`'reset' im Zustand 'error' gerade nicht
  möglich`“. Dann zuerst `undock`.
- Beim Ablauf-Ende hängt der Folgezustand vom Standort ab: im Dock `docked`, frei `idle`,
  in der Zone `error`.
- **Pause** hält nichts mitten in einer Bewegung an: Der laufende Schritt (Laufen, Greifer,
  eine schon gestartete Bandfahrt) wird fertig, danach wartet der Ablauf vor dem nächsten
  Schritt. Wartezeiten (z. B. 10 s „Stütze aufs Band stellen“, 30 s Pause) stehen in der
  Pause still. Im Band-Strang startet in der Pause keine neue Bandfahrt. Sofort anhalten:
  `stop` oder `estop`.
- Der Ablauf ist eine **Schrittliste** (`Station._build_plan()`), die beim Start feststeht.
  Daraus zeigt das Frontend vorherigen, aktuellen und nächsten Schritt; `station/state`
  enthält `step_index`, `step_total`, `prev_step`, `next_step`, `cycle` und `holding`.

### 6.3 Ereignisse des Greifers

Meldet der Greifer `offline` oder `connection_lost` (bzw. `part_lost`, aber nur bei
aktivem Teilesensor), löst die Station `fault` aus: Alles wird gestoppt, Zustand `error`.

---

## 7. Befehle

Befehle kommen über die Tastatur (Terminal von `main.py`) oder als Text per MQTT auf
`station/cmd`. Kommas werden wie Leerzeichen behandelt (`goto 2.0,-0.5` = `goto 2.0 -0.5`).

### Betrieb

| Befehl | Erlaubt in | Wirkung |
|---|---|---|
| `start` [`alles`\|`liefern`\|`holen`] | `idle` | Ablauf starten (Standard: `alles`) |
| `liefern` | `idle` | nur Ausliefern |
| `holen` | `idle` | nur Zurückholen |
| `pause` | `running` | Ablauf nach dem laufenden Schritt anhalten |
| `resume` | `paused` | Ablauf weiterlaufen lassen |
| `stop` | Busy-Zustände (auch `paused`) | laufenden Ablauf oder Befehl sauber abbrechen |
| `estop` | überall | Not-Aus: SID, Band und Achse sofort stoppen |
| **NOT-AUS-Knopf / Leertaste** (Frontend) | überall | wie `estop` |
| `reset` | `error`, `estop` | nach `idle` oder `docked` (siehe 6.2); fehlt die Startpose: wie `init` |
| `init` | `error` ohne Startpose | Startpose neu messen (passiert auch automatisch, sobald die Ortung trackt) |
| `status` | überall | Zustand als Dictionary ausgeben |
| `help` | überall (nur Tastatur) | Befehlsübersicht |
| `quit` / `q` / `exit` | nur Tastatur | Programm beenden |
| `q` / ESC (nur `--cv-window`) | – | Programm beenden |

Im Frontend gibt es dafür Knöpfe mit Tastenkürzeln (siehe 10.1).

### Einzelbefehle zum Testen

| Befehl | Erlaubt in | Wirkung |
|---|---|---|
| `goto x y [θ]` | `idle` | Punkt anfahren, optional mit Zielwinkel |
| `approach x y [winkel] [abstand]` | `idle` | Punkt aus fester Richtung anfahren (Standard 0°, 0,5 m) |
| `dock` | `idle` | Andocken |
| `undock` | `docked`, `error` (in Zone) | gerade rückwärts aus der Zone |
| `convey <s>` | `idle`, `docked` | Band für s Sekunden fahren |
| `detect` | `idle`, `docked` | eine YOLO-Erkennung |
| `home` | `idle`, `docked` | Referenzfahrt der Achse |
| `arm_to <cm>` | `idle`, `docked` | Achse auf Höhe in cm (0 … 44 cm) |
| `grip` / `release` | `idle`, `docked` | Greifer schließen / öffnen |

---

## 8. MQTT-Schnittstelle

Broker `192.168.2.1:1883`. Jeder Client braucht eine eigene Client-ID, sonst trennt der
Broker die Verbindungen gegenseitig. Alle Clients verbinden sich im Hintergrund und nach
einem Abbruch selbst neu; ein fehlender Broker hält kein Programm an.

| Client-ID | Programm |
|---|---|
| `station_main`, `station_conveyor`, `station_arm`, `station_vision`, `station_frontend` | `main.py` (Station, Band, Greifer, YOLO, Ortungsbild) |
| `station_tracker` | `aruco_tracker_calibrated.py` |
| `trainer_frontend` | `trainer.py` (Ortungsbild, Pose, Status) |
| zufällig | Posen-Empfänger in `navigate.py` und den `collect_*.py` |

### 8.1 Station

| Topic | Richtung | Inhalt |
|---|---|---|
| `station/cmd` | → Station | Befehl als Text, z. B. `start`, `goto 2.0 0.3` |
| `station/state` | Station → | JSON, retained, nach jedem Zustandswechsel und Schritt. Last Will: `{"state": "offline"}` |

Beispiel `station/state`:

```json
{
  "state": "running", "phase": "Ausliefern 2/3",
  "step": "SID läuft in die Dockingstation", "band": "Stütze 3 von Hand aufs Band stellen",
  "message": "", "location": "FREE", "pose": [1.21, -0.25, 179.8],
  "conveyor": "idle", "arm": "idle", "gripper": "open", "detections": 1,
  "tracker": "tracking", "prev_state": "idle",
  "cycle": "running", "holding": false,
  "step_index": 7, "step_total": 68,
  "prev_step": "Greifer öffnet sich für die Übergabe", "next_step": "Greifer schließt – SID übernimmt die Stütze",
  "devices_ok": {"conveyor": true, "arm": true, "detector": true},
  "time": 1790312345.6
}
```

`location` ist `FREE`, `IN_ZONE` oder `DOCKED`. `cycle` ist `running`, `done`, `aborted`,
`failed` oder leer (noch kein Ablauf); `holding` ist `true`, wenn der Ablauf in der Pause
tatsächlich stillsteht.

### 8.2 Geräte

| Topic | Richtung | Inhalt |
|---|---|---|
| `station/tracker/robot/pose` | Ortung → | JSON `{"seq", "t", "x", "y", "theta"}` pro Bild mit gültiger Pose (nicht retained) |
| `station/tracker/status` | Ortung → | retained: `teaching`, `tracking`, `lost`, `offline` |
| `station/tracker/cmd` | → Ortung | `reteach`: Koordinatensystem neu einlernen |
| `station/tracker/preview` | Ortung → | nur mit `--preview`: Debugbild als JPEG (960 px breit, 5 Bilder/s, nicht retained), fürs Frontend |
| `station/arm/cmd/axis` | → Greifer | `home` oder Zielposition in Schritten |
| `station/arm/cmd/gripper` | → Greifer | Servozeit in µs (optional mit ` check`) |
| `station/arm/cmd/stop` | → Greifer | `stop` |
| `station/arm/cmd/status` | → Greifer | `status` |
| `station/arm/cmd/config` | → Greifer | Parameter `key=value …`, retained |
| `station/arm/state` | Greifer → | JSON (state, steps, homed, gripper, part …), retained |
| `station/arm/event` | Greifer → | JSON `{"level", "code", "info"}` |
| `station/arm/heartbeat` | Greifer → | alle 3 s |
| `station/conveyor` | → Band | Fahrdauer in s oder `stop` |
| `station/conveyor/status` | Band → | `idle`, `running`, `offline` (retained) |
| `station/conveyor/error` | Band → | abgelehnte Befehle |
| `station/vision/detections` | Vision → | JSON mit allen Erkennungen einer Aufnahme |
| `station/vision/status` | Vision → | `online` / `offline` (retained) |

### 8.3 Frontends (HTTP, nur lokal)

Beide Webserver laufen im jeweiligen Programm (nur Python-Standardbibliothek) und lauschen
standardmäßig nur auf `127.0.0.1`.

| Adresse | Programm | Wichtigste Endpunkte |
|---|---|---|
| `http://localhost:8080/` | Station | `/api/events` (Zustand als Server-Sent Events, 5×/s), `/api/cmd` (start, pause, stop, ack, estop, quit), `/api/hotkeys`, `/feed/<name>.mjpg` |
| `http://localhost:8081/` | Bewegungstraining | `/api/events` (4×/s, inkl. neuer Ausgabezeilen), `/api/cmd` (collect, train, stop, activate, quit), `/feed.mjpg`, `/plots/<name>.png` |

Mit `host: 0.0.0.0` in `config/frontend.yaml` ist das Stations-Frontend auch von anderen
Rechnern erreichbar – dann kann jeder im Netz die Station bedienen.

---

## 9. Die Geräte im Detail

### 9.1 Navigation (`scripts/navigate.py`)

**Prinzip:** gelernte Bewegungsmodelle, geplant mit einem vorausschauenden Regler, der
nach jedem Schritt neu plant (Receding Horizon).
Jeder Schritt läuft so ab:

1. Pose im Stand messen. Gemittelt wird 3 bis 4 s nach dem Stopp, im selben Zeitfenster
   wie bei der Datensammlung, sonst passt das Modell nicht.
2. Befehl wählen:
   - **Grobphase** (weiter als 0,30 m vom Ziel): eine Drehung, dann ein langer gerader Lauf
     bis 0,10 m vor das Ziel.
   - **Feinphase:** Beam-Search über alle Lauf-Primitive (Horizont 3, Strahlbreite 60).
     Kosten = erwarteter quadratischer Zielfehler plus Strafen für Schritte und Zeit.
   - **Endausrichtung** (falls ein Zielwinkel verlangt ist): gezielte Drehung aus dem
     Drehmodell.
3. Nur den ersten Befehl ausführen, anhalten, zurück zu 1.

Ziel erreicht: 5 cm Position, 5° Winkel. Nach 25 Schritten oder wenn kein Schritt mehr
verbessert, endet der Befehl mit `FAILED`.

**Modelle:**

- `dynamics_model.joblib`: pro Aktion (forward, backward, left, right, fwd_left …, yaw_*)
  eine Physik-Grundlinie `v · softplus(T − Totzeit) + b` plus ein Gaußprozess auf den
  Residuen. Liefert Verschiebung und Drehung inkl. Unsicherheit.
- `rotation_model.joblib`: präzise Drehungen (`single`, `flick`, `flick_pause`). Fehlt
  es, dreht der SID mit `yaw_left` / `yaw_right`.
- Die Bogen-Aktionen `arc_*` sind ausgeschlossen (zu ungenau, 5 bis 12 cm).

**Anfahren aus fester Richtung (`approach`)** läuft in zwei Hälften:
`approach_pre` fährt zur Vorposition (0,5 m vor dem Punkt, 4 cm seitlich / 4° Toleranz,
gelockert 12 cm / 12° mit Warnung). `approach_final` fährt danach nur noch vorwärts, in
höchstens 2 Schritten. Gilt als Erfolg bis 15 cm längs, 10 cm seitlich, 15°.

**Docken:**

1. Dock-Vorposition (1.20, −0.25, 180°) anfahren und ausrichten. Klappt das nicht, 0,15 m
   gerade zurück und neu versuchen.
2. Drehsperre an. **Ein** gerader Vorwärtslauf bis zur Dockmitte. Die Dauer ergibt sich
   aus der gemessenen Entfernung: `T = Totzeit + Strecke / Geschwindigkeit` (aus dem Modell
   geschätzt, oder fest über `DOCK_SPEED` / `DOCK_DEADTIME`).
3. Erfolgreich, wenn höchstens 25 cm zu kurz und höchstens 10 cm seitlich. Zu weit ist in
   Ordnung, der Puffer im Dock fängt das ab.
4. Sicherheitsstopp bei mehr als 20° (im Stand) bzw. 30° (während der Fahrt)
   Winkelabweichung oder mehr als 20 cm seitlicher Abweichung.

**Ausdocken:** ein gerader Rückwärtslauf bis x ≥ 1,30 m (maximal 3 Läufe). Erst danach ist
Drehen wieder erlaubt.

**Ergebnisse (`NavResult.status`):**

| Status | Bedeutung | Folge im Automaten |
|---|---|---|
| `SUCCESS` | Ziel erreicht | weiter |
| `FAILED` | Ziel nicht genau genug erreicht | im Ablauf je nach `exact` Warnung oder Fehler |
| `ABORTED` | durch `stop` / `estop` abgebrochen | `cycle_aborted` bzw. `nav_aborted` |
| `SAFETY_STOP` | Sicherheitsverletzung, SID gestoppt | Fehler |
| `REJECTED` | Befehl unzulässig, nichts bewegt | Einzelbefehl: zurück nach `idle`; im Ablauf: Fehler |
| `ERROR` | unerwarteter Fehler | Fehler |

**Überwachung während jeder Bewegung** (alle 50 ms):

- Keine neue Pose seit 1,5 s → sofort Stopp („Tracker liefert während der Bewegung keine
  aktuelle Pose!“).
- SID mehr als 4 m von der Startpose entfernt → Stopp.
- Zonen- bzw. Andockbahn-Wächter (siehe Abschnitt 4).

**Befehle an den SID** (`lite3_commander.py`): Heartbeat alle 0,5 s, Steuermodus MANUAL,
Bewegungsmodus MOVE, Achsen X / Y / Yaw mit ±32767. Kleine Werte ignoriert der SID
(Totzonen X 6553, Y 12553, Yaw 9553). Es werden keine Sprünge, Gangartwechsel oder
Aktions-Skills benutzt.

### 9.2 SID-Ortung (`scripts/aruco_tracker_calibrated.py`)

- Eigener Prozess. `main.py` und `trainer.py` starten ihn, falls noch keiner läuft (Prüfung
  über `pgrep -f aruco_tracker`), und zwar ohne Fenster mit `--headless --preview`: Das
  Debugbild geht als JPEG per MQTT (`station/tracker/preview`, 960 px, 5 Bilder/s) ins Frontend.
  Wer die Ortung gestartet hat, beendet sie auch wieder. Mit `main.py --cv-window` läuft sie
  wie früher mit eigenem Fenster.
- Findet die Kamera über den Gerätenamen (`arducam`), nimmt MJPG in maximaler Auflösung
  (1920×1080) und entzerrt nur die gefundenen Markerecken (Fisheye-Kalibrierung vom 25.09.,
  26 Bilder, RMS 0,377 px).
- **Einlernen:** Über 30 Bilder werden die Referenzmarker 1 bis 3 gemittelt, Status
  `teaching`. Verschieben sie sich später (mehr als 8 px in 5 aufeinanderfolgenden Bildern,
  z. B. weil die Kamera bewegt wurde), wird neu eingelernt.
- Eine Pose wird nur gesendet, wenn das Koordinatensystem gültig ist und der SID im
  aktuellen Bild erkannt wurde. Nach 0,5 s ohne SID wechselt der Status auf `lost`.
- Tasten im Fenster (falls mit Fenster gestartet): `r` neu einlernen, `q`/ESC beenden.
  Ohne Fenster neu einlernen: `mosquitto_pub -h 192.168.2.1 -t station/tracker/cmd -m reteach`.
- Aufruf von Hand: `python scripts/aruco_tracker_calibrated.py [--headless] [--preview] [--resolution BxH]`.

### 9.3 Greifer (`scripts/arm_controller.py`, `scripts/arm_config.py`)

Der ESP32 führt Befehle nur aus. Alle Grenzen prüft der PC:

| Parameter | Wert |
|---|---|
| Achse | 200 Schritte/Umdrehung, 0,8 cm/Umdrehung → 250 Schritte/cm |
| Achsbereich | 0 (Endschalter unten) … 11000 Schritte (= 44 cm). **Oben gibt es keinen Schalter**, `AXIS_MAX_STEPS` ist die einzige Absicherung |
| Geschwindigkeit | 400 µs/Schritt → 2500 Schritte/s (0 → max ≈ 4,5 s) |
| Zange | 980 µs = zu, 1600 µs = offen, Bewegungszeit 800 ms |
| Teilesensor | **aus** (`USE_PART_SENSOR = False`): Greifen gilt ohne Prüfung als erfolgreich |
| Heartbeat | alle 3 s, nach 10 s gilt der ESP32 als nicht erreichbar |
| Befehl mit `wait=True` | wartet höchstens 20 s auf die Fertigmeldung |

Die Bewegungsparameter gehen beim Verbinden **retained** an den ESP32
(`station/arm/cmd/config`), er bekommt sie also auch nach einem eigenen Neustart.
`limit_reached` beim Herunterfahren ist normal (min-Position = Endschalter) und kein Fehler.

### 9.4 Fließband (`scripts/conveyor_controller.py`)

- `drive(s, wait=True)`: sendet die Fahrdauer. Das Fahrtende ist erkannt, sobald der Status
  von `running` wieder auf `idle` wechselt. Timeout: Fahrdauer + 5 s.
- Fahrbefehle werden nur gesendet, wenn der Status `idle` ist.
- Eine volle Fahrt vom Bandanfang zur Dockingstation dauert 18 s.
- Beim Trennen wird eine laufende Fahrt gestoppt.

### 9.5 Stützenerkennung (`scripts/F_stilt_erkennung_YOLO.py`)

- Modell `models/best.pt` (Ultralytics YOLO), Eingabegröße 640, Konfidenz ≥ 0,5, CPU.
- Kamera: erster `/dev/videoN`, der YUYV-Bilder liefert (Prüfung mit `v4l2-ctl`).
  Mit `NAME_HINT` lässt sich eine bestimmte Kamera festlegen.
- Vor jeder Erkennung werden 3 Bilder verworfen, damit das Bild aktuell ist.
- Aufnahmen mit Treffern werden auf dem Desktop in `ir_detections/` gespeichert
  (roh + annotiert). Leere Bilder speichert die Station bei der Dauersuche nicht.
- Ergebnis pro Stütze: `id`, `label`, `confidence`, Box `x1 y1 x2 y2`, Mitte `cx cy`,
  `width`, `height` (Pixel), sortiert von links nach rechts.
- Direkt gestartet (`python scripts/F_stilt_erkennung_YOLO.py`) öffnet sich eine
  Live-Ansicht: Leertaste erkennt, `c` löscht die Boxen, `n` normiert die Vorschau,
  `q` beendet.

---

## 10. Anzeige und Logging

**Terminal:** nur leicht verständliche Stationsmeldungen sowie Warnungen und Fehler der
Geräte.

| Symbol | Farbe | Bedeutung |
|---|---|---|
| `══════ TEXT ══════` | lila | neuer Abschnitt (AUSLIEFERN, PAUSE …) |
| ▶ | blau | etwas beginnt |
| ✔ | grün | etwas ist fertig |
| ⏳ | gelb | Station wartet (auch auf den Menschen!) |
| • | weiß | Information |
| ✖ | rot | Fehler |
| `!` | gelb/rot | Warnung/Fehler eines Geräts |

**Logdatei** `logs/station_<Datum>_<Zeit>.log` (Bewegungstraining: `logs/trainer_…`): alles auf DEBUG-Ebene, auch jeder
Planungsschritt der Navigation und jede MQTT-Nachricht der Geräte. Bei Fehlern immer zuerst
hier nachsehen.

**Frontend:** siehe 10.1. Nur mit `--cv-window` gibt es stattdessen die alten Fenster:

- **Kamerafenster „YOLO-Kamera“:** Live-Bild der IR-Kamera mit den letzten Erkennungen und
  oben Zustand, Phase, SID-Schritt, Band-Status und Anzahl erkannter Stützen.
- **Ortungsfenster „Arducam ArUco Tracking“:** Kamerabild mit Koordinatenachsen,
  Robotermarker und Pfeil in Blickrichtung, FPS und Status.

**Bewegungslogs** `data/nav_logs/nav_log.csv` / `nav_rotation_log.csv` (Simulation: `*_sim.csv`): jede
ausgeführte Bewegung im Format der Trainingsdaten, zum Nachtrainieren (siehe 12).

### 10.1 Frontend (Bedienoberfläche)

Das Frontend startet als Erstes und läuft immer, auch wenn Broker, Geräte oder die ganze
Station nicht starten (siehe 3.4). `main.py` startet dazu einen kleinen Webserver (`scripts/station_frontend.py`, nur
Standardbibliothek) und öffnet die Oberfläche im Vollbild (Chromium oder Firefox im
Kiosk-Modus mit eigenem Profil; beenden mit Alt+F4). Adresse: `http://localhost:8080/`
(einstellbar in `config/frontend.yaml`). Die Station läuft weiter, wenn das Fenster
geschlossen wird; die Adresse einfach neu öffnen.

| Bereich | Inhalt |
|---|---|
| Kopfzeile | Zustand (groß, farbig; `FAILED` = Station nicht gestartet), Abschnitt, letzte Meldung, Zusammenfassung „Systeme x/y ok“ |
| Systeme (rechts oben) | je Teilsystem ein Punkt (grün läuft, gelb eingeschränkt, rot läuft nicht, grau nicht nötig/startet) mit Grund: MQTT-Broker, SID-Ortung, SID (Lite3) mit Pose, Fließband, Greifer, YOLO-Kamera |
| Kamerabild links | „Kamera Greifarm“, Standard: YOLO-Kamera mit erkannten Stützen |
| Kamerabild rechts | Debugbild der SID-Ortung (Marker, Achsen, SID-Pfeil) per MQTT `station/tracker/preview` |
| Ablauf | Fortschrittsbalken aller Schritte, vorheriger / **aktueller** / nächster Schritt, Band-Strang, Zustandskette des Automaten (aktuell + vorher) |
| Rechts unten | Fehlermeldungen (Fehler, Gerätewarnungen, Systemausfälle; quittierbar) und alle Stationsmeldungen |
| Unten | Ablaufwahl (Alles / Liefern / Holen), START, PAUSE/WEITER, STOP, QUITTIEREN, NOT-AUS |

| Knopf | Standardtaste | Wirkung |
|---|---|---|
| START | F2 | `start <Ablaufwahl>` (nur in `idle`) |
| PAUSE / WEITER | F3 | `pause` bzw. `resume` |
| STOP | F4 | `stop` |
| QUITTIEREN | F8 | alle Fehlermeldungen quittieren; in `error` / `estop` zusätzlich `reset` |
| NOT-AUS | Leertaste | `estop` (geht immer, auch bei offenem Einstellungsdialog) |
| BEENDEN | F10 | Programm beenden (mit Rückfrage; nochmal F10 oder Klick bestätigt, Enter bricht ab) |

Die Tasten lassen sich über **⚙ Tasten** im Frontend ändern (Taste drücken, speichern)
oder direkt in `config/hotkeys.yaml`. Knöpfe, die im aktuellen Zustand nicht erlaubt sind,
sind ausgegraut; ein Tastendruck darauf zeigt nur einen Hinweis. Bricht die Verbindung zur
Station ab, legt sich „KEINE VERBINDUNG“ über die Anzeige; ein Kamerabild ohne neue Bilder
seit 2 s zeigt „KEIN BILD“.

Die Quellen der Kamerabilder stehen in `config/frontend.yaml` (`yolo`, `camera:<Nummer|Name>`
für eine eigene USB-Kamera, `mqtt:<Topic>`). Das Ortungsbild gibt es nur, wenn
`aruco_tracker_calibrated.py` mit `--preview` läuft (startet `main.py` die Ortung selbst,
passiert das automatisch).

> Der NOT-AUS im Frontend ist ein Software-Not-Aus wie `estop`, kein Ersatz für einen
> Hardware-Not-Aus.

---

## 11. Konfiguration: wo steht was?

Fast alle Parameter sind Konstanten am Anfang der jeweiligen Datei. Nur das Frontend hat
YAML-Dateien in `config/`.

| Was | Datei | Konstanten |
|---|---|---|
| Positionen, Wartezeiten, Stützenzahl | `main.py` | `PLACE_POSITIONS`, `PAUSE_POSITION`, `DROP_POSITION`, `*_S`, `BACK_OFF_M` |
| Broker der Station, Topics | `main.py` | `BROKER`, `PORT`, `TOPIC_*` |
| SID-IP, Zone, Dock, Toleranzen, Planer | `scripts/navigate.py` | `ROBOT_IP`, `SAFETY_ZONE_X`, `DOCK_*`, `APPROACH_*`, `POS_TOLERANCE` … |
| Greifer | `scripts/arm_config.py` | alle; neu laden per `config` in `arm_console.py` |
| Fließband | `scripts/conveyor_controller.py` | `BROKER`, `TOPIC_*`, `WAIT_MARGIN_S` |
| YOLO / IR-Kamera | `scripts/F_stilt_erkennung_YOLO.py` | `MODEL_FILE`, `CONF_THRESHOLD`, `NAME_HINT`, `WIDTH`/`HEIGHT` |
| Ortung / Kalibrierung | `scripts/aruco_tracker_calibrated.py` | `CALIB_*`, `MARKER_LENGTH_M`, `ROBOT_MARKER_CAMERA_DISTANCE_M`, `PREVIEW_*` |
| Frontend | `config/frontend.yaml`, `config/hotkeys.yaml` | Adresse/Port, Browser, Kamerabilder, Tastenkürzel |
| Ablage von Modellen und Daten | `scripts/paths.py` | `MODELS`, `DATASETS`, `NAV_LOGS`, `PLOTS` … |
| Bewegungstraining | `scripts/trainer_server.py`, `trainer.py` | Port (8081), `STOP_TIMEOUT_S`; Policies kommen aus den `collect_*.py` |

**Achtung:** Die Broker-Adresse steht in mehreren Dateien (`main.py`, `navigate.py`,
`arm_config.py`, `conveyor_controller.py`, `F_stilt_erkennung_YOLO.py`,
`aruco_tracker_calibrated.py`, `collect_common.py`). Bei einem Wechsel alle anpassen.

---

## 12. Laufmodelle: Daten sammeln und trainieren

Die Laufmodelle müssen neu trainiert werden, wenn sich das Laufverhalten des SID ändert
(Firmware, Untergrund, Beladung) oder die Ortung neu kalibriert wurde. Die Daten vor der
aktuellen Kalibrierung liegen in `data/archiv/datasets_falsch_kalibriert/` und dürfen
nicht mehr benutzt werden.

### 12.1 Bewegungstraining (Programm mit Frontend)

Desktop-Symbol **Bewegungstraining** oder `python trainer.py` (Frontend im Vollbild,
`http://localhost:8081/`). Es ist unabhängig von der Station und startet die SID-Ortung mit
Vorschaubild selbst, falls sie nicht läuft.

| Bereich | Inhalt |
|---|---|
| links oben | Bild der SID-Ortung (Arducam, ArUco) mit aktueller Pose |
| links unten | Ausgabe des laufenden Skripts (live) |
| rechts oben | laufender Lauf: Fortschritt Sample i/N, aktueller Befehl, letztes Ergebnis, Abstand zum Startpunkt (Balken bis zum Not-Aus-Radius, Strich = Arbeitsbereich) |
| Reiter **Sammeln** | Trainings-Policy wählen (Geradlinig, Gemischt, Drehungen, Feine Schritte; bei Feine Schritte die Gruppen), Anzahl Samples, Simulation, SAMMELN STARTEN |
| Reiter **Trainieren** | Laufmodell oder Drehmodell, Datensätze ankreuzen, TRAINIEREN; danach den Kandidaten ÜBERNEHMEN |
| Reiter **Daten** | alle Datensätze mit Zeilenzahl und Stand, Diagramme aus `data/plots` (anklicken = groß) |
| unten | BEENDEN (F10), **STOPP (Leertaste)**: hält einen laufenden Lauf sofort an |

Sicherheit: Mit dem echten Roboter wird nur gesammelt, wenn die Station (`main.py`) **nicht**
läuft (beide würden den SID steuern) und die Ortung den SID verfolgt; vor dem Start kommt eine
Rückfrage. Immer nur ein Lauf gleichzeitig. STOPP wirkt wie Strg+C: Das Skript stoppt den
Roboter und schließt die Dateien. Simulation (ohne Roboter, Dateien `*_sim.csv`) geht immer.

Trainieren erzeugt einen **Kandidaten** in `models/kandidaten/`. Erst ÜBERNEHMEN sichert das
bisherige Modell in `models/archiv/` und kopiert den Kandidaten nach `models/`. Er gilt ab dem
nächsten Start der Station.

Hinweis zur Ortung: Wer sie startet, beendet sie auch wieder. Hat das Bewegungstraining die
Ortung gestartet, endet sie mit ihm, auch wenn die Station inzwischen läuft (und umgekehrt).
Deshalb nicht beide Programme gleichzeitig betreiben.

### 12.2 Von Hand (Terminal)

**1. Daten sammeln** (SID frei im Raum, Ortung läuft, Station aus):

```bash
python scripts/collect_linear.py   [--samples N] [--sim]   # vor/zurück/seitlich
python scripts/collect_mixed.py    [--samples N] [--sim]   # Kombinationen
python scripts/collect_rotation.py [--samples N] [--sim]   # Drehungen
python scripts/collect_fine.py     [--samples N] [--sim] [--groups yaw,lateral,sideyaw,longitudinal]
```

Die Skripte wählen nur Befehle, die grob geschätzt im Arbeitsbereich (1 m um den Start)
enden. Ab 1,6 m Entfernung stoppen sie hart. Entfernt sich der SID mehr als 0,5 m, laufen sie
ihn ohne Drehen zurück. Diese Rückläufe landen in `*_return_log.csv` und sind ebenfalls
Trainingsdaten. Alles wird in `data/datasets/` angehängt.

**2. Trainieren:**

```bash
python scripts/train_dynamics.py --csv linear_dataset.csv mixed_dataset.csv rotation_dataset.csv \
    linear_return_log.csv mixed_return_log.csv rotation_return_log.csv nav_log.csv
python scripts/train_rotation.py --csv rotation_dataset.csv
```

Ausgabe: `models/dynamics_model.joblib`, `models/rotation_model.joblib` (überschreibt das
aktive Modell; mit `--out` woanders hin), Diagramme in `data/plots/` und eine
Kreuzvalidierung in der Konsole. Samples mit unruhiger Messung (Streuung > 2 cm bzw. > 3°)
werden verworfen. Aus `fine_dataset.csv` nimmt `train_dynamics.py` nur die Einzelimpulse
(`kind = pulse`).

**3. Prüfen:** `python scripts/check_heading.py linear_dataset.csv` prüft, ob der gemessene
Winkel zur tatsächlichen Laufrichtung passt (`data/plots/heading_check.png`). Ein konstanter
Versatz deutet auf einen verdreht aufgeklebten Marker hin, ein Muster über θ auf eine
falsche Kalibrierung.

---

## 13. Test- und Hilfswerkzeuge

| Aufruf | Zweck |
|---|---|
| `python main.py --sim-nav` | Ablauf mit simuliertem SID, alle anderen Geräte echt |
| `python trainer.py` / `./start_trainer.sh` | Bewegungstraining mit Frontend (siehe 12.1) |
| `python main.py --no-window` | kein Fenster öffnen (z. B. über SSH); Frontend trotzdem unter der Adresse erreichbar |
| `python main.py --cv-window` | alte OpenCV-Fenster statt Frontend |
| `./start_station.sh` | Start wie über das Desktop-Symbol (läuft die Station schon: nur Frontend öffnen); Argumente gehen an `main.py` |
| `./station_desktop.sh icon` | Desktop-Symbole „Station“ und „Bewegungstraining“ anlegen |
| `./station_desktop.sh autostart-on` / `autostart-off` / `status` | Autostart nach dem Anmelden (mit automatischer Anmeldung = beim Hochfahren) |
| `python scripts/navigate.py --sim` | Navigation allein in der Simulation, interaktiv |
| `python scripts/navigate.py --sim --goals 2.0,0.3 d u` | Ziel, docken, ausdocken |
| `python scripts/navigate.py` | Navigation allein am echten SID (mit eigener START-Abfrage) |
| `python scripts/arm_console.py` | Greifer von Hand: `home`, `max`, `cm 12`, `open`, `close`, `state`, `config` |
| `python scripts/conveyor_console.py` | Band von Hand: Zahl = Sekunden, `s` = Stopp |
| `python scripts/F_stilt_erkennung_YOLO.py` | IR-Kamera + YOLO allein |
| `python scripts/aruco_tracker_calibrated.py` | Ortung allein |
| `mosquitto_sub -h 192.168.2.1 -t 'station/#' -v` | alle MQTT-Nachrichten mitlesen |
| `mosquitto_pub -h 192.168.2.1 -t station/cmd -m start` | Befehl per MQTT senden |

---

## 14. Bekannte Einschränkungen

- **Roboterarm fehlt:** Die Stütze muss von Hand aufs Band gestellt werden (10 s nach der
  Erkennung).
- **Teilesensor am Greifer ist aus:** Ob eine Stütze wirklich gegriffen wurde, wird nicht
  geprüft.
- **Not-Aus in der Zone:** Wird `estop` ausgelöst, während der SID in der Zone, aber nicht
  im Dock steht, ist weder `reset` noch `undock` möglich. Ausweg: Programm beenden und neu
  starten. Der SID startet dann in `error`, danach `undock` und `reset`.
- **Station und Bewegungstraining nicht gleichzeitig:** Beide steuern den SID über denselben
  UDP-Port. Das Training sperrt das Sammeln, solange die Station läuft; umgekehrt prüft die
  Station das nicht. Außerdem beendet jedes Programm beim Schließen die Ortung, die es
  selbst gestartet hat.
- **YOLO-Kamera nachträglich anstecken** wird nicht erkannt: `main.py` neu starten.
- **NOT-AUS im Frontend** ist ein Software-Not-Aus (`estop`), kein Ersatz für einen
  Hardware-Not-Aus.
- **Frontend-Bild der Ortung** braucht die Ortung mit `--preview`. Läuft eine von Hand ohne
  diese Option gestartete Ortung, bleibt das Bild leer („KEIN BILD“).
- **`start` nur aus `idle`:** Steht der SID im Dock, erst `undock`.
- **Sicherheitsradius:** Ziele weiter als 4 m von der Startpose werden abgelehnt.
- **`requirements.txt` ist unvollständig:** Es fehlen `ultralytics` (inkl. `torch`),
  `joblib`, `scikit-learn`, `scipy`, `pandas` und `matplotlib`. In `.venv` sind sie
  installiert.
- Die in `CLAUDE.md` geplante Struktur (`core/`, `devices/`, `orchestration/`) ist noch nicht
  umgesetzt. Alle Geräte liegen als Module in `scripts/`; YAML-Konfiguration gibt es bisher
  nur für das Frontend (`config/`).
- Die Firmware der ESP32 (Greifer: `Arm_MQTT.ino`, Fließband) liegt nicht in diesem
  Repository.
