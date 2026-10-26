# Station – Inbetriebnahme und Troubleshooting

Diese Anleitung führt vom ausgeschalteten Aufbau bis zum laufenden Ablauf. Die technischen
Hintergründe stehen in [DOKUMENTATION.md](DOKUMENTATION.md).

> **Sicherheit:** Der SID bewegt sich nach `start` selbstständig. Während des Betriebs
> niemand im Fahrbereich, außer um die Stütze aufs Band zu stellen. Not-Aus jederzeit mit
> dem **NOT-AUS-Knopf / der Leertaste** im Frontend oder mit `estop` im Terminal.

---

## Teil A – Einmalige Einrichtung des Steuer-PCs

Nur nötig auf einem neuen PC oder nach einer Neuinstallation.

```bash
cd ~/sid1walz40
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# fehlen in requirements.txt, werden aber gebraucht:
pip install ultralytics joblib scikit-learn scipy pandas matplotlib
sudo apt install v4l-utils mosquitto-clients
```

Prüfen:

- [ ] In `models/` liegen `dynamics_model.joblib`, `rotation_model.joblib` und `best.pt` (YOLO-Modell).
- [ ] Der PC erreicht beide Netze:
  `ping 192.168.2.1` (MQTT-Broker) und `ping 192.168.1.120` (SID).
- [ ] Der Benutzer darf auf die Kameras zugreifen (Gruppe `video`): `ls -l /dev/video*`.

Desktop-Symbol und (optional) Autostart einrichten:

```bash
./station_desktop.sh icon            # Symbole „Station“ und „Bewegungstraining“ auf den Schreibtisch
./station_desktop.sh autostart-on    # optional: Station startet nach dem Anmelden von selbst
./station_desktop.sh status          # zeigt Symbol, Autostart und automatische Anmeldung
```

Mit automatischer Anmeldung (Einstellungen > Benutzer) bedeutet Autostart: Start beim
Hochfahren. Ausschalten mit `./station_desktop.sh autostart-off`.

---

## Teil B – Aufbau vor jedem Betrieb

### B1 Raum und Marker

- [ ] Referenzmarker ID 1, 2, 3 liegen an ihrem Platz auf dem Boden, flach, sauber und
  nicht verdeckt. **Nicht verschieben**, sonst ändert sich das Koordinatensystem.
- [ ] Marker ID 0 sitzt fest auf dem SID.
- [ ] Die Arducam an der Decke ist nicht verstellt.
- [ ] Fahrbereich frei, die drei Plätze (x = 2,0 m; y = −0,6 / 0 / +0,6) sind leer.
- [ ] Dockingstation und Band stehen an ihrem Platz, der Weg vom Band ins Dock ist frei.

### B2 Geräte einschalten

1. MQTT-Broker (192.168.2.1) läuft.
2. Fließband-ESP32 und Greifer-ESP32 einschalten. Der Greifer öffnet beim Booten die Zange
   (`OPEN_GRIPPER_ON_BOOT`). **Vorher keine Stütze im Greifer lassen.**
3. Beide Kameras am PC einstecken (Arducam und IR-Kamera).
4. SID einschalten, aufstehen lassen und **außerhalb der Sicherheitszone** hinstellen,
   also x > 1,3 m, am besten bei etwa (2,0; 0), Marker frei sichtbar.
   Alle Ziele müssen innerhalb von 4 m um diese Startpose liegen.

Schnelltest, ob alle Geräte am Broker sind:

```bash
mosquitto_sub -h 192.168.2.1 -t 'station/#' -v -W 5
```

Erwartet: `station/conveyor/status idle`, `station/arm/state {...}`,
`station/arm/heartbeat ...` (alle 3 s). Falls die Ortung schon läuft, auch
`station/tracker/status tracking`.

### B3 Stützen bereitlegen

- `start` / `liefern`: Die erste Stütze steht auf der **Erkennungsfläche** im Bild der
  IR-Kamera. Die anderen beiden liegen bereit.
- `holen`: Die drei Stützen stehen auf ihren Plätzen.

---

## Teil C – Station starten

Doppelklick auf das Symbol **Station** auf dem Schreibtisch (oder Autostart). Es öffnen
sich ein Terminalfenster mit den Meldungen und das Frontend im Vollbild. Läuft die Station
schon, öffnet das Symbol nur das Frontend wieder.

Von Hand geht es so:

```bash
cd ~/sid1walz40
source .venv/bin/activate
python main.py
```

Was passiert (bei jeder Meldung kurz mitlesen):

| Meldung | Bedeutung / was tun |
|---|---|
| `══════ STATION STARTET ══════` | Programm läuft, Logdatei wird genannt |
| `Frontend läuft: http://localhost:8080/` | Frontend ist da; rechts oben „Systeme“ zeigt, was läuft und was nicht |
| `Starte SID-Ortung ...` | Ortung startet ohne eigenes Fenster, ihr Bild erscheint im Frontend (oder: „läuft bereits“) |
| `YOLO-Kamera bereit (/dev/videoN)` | IR-Kamera und Modell geladen |
| `Alle Systeme bereit.` bzw. `Nicht bereit: …` (nach ca. 8 s) | Bei `Nicht bereit` siehe Systeme im Frontend und Troubleshooting |
| `Warte auf die SID-Ortung ...` → `SID-Ortung verfolgt den SID` | Referenzmarker eingelernt, SID erkannt |
| `Messe die Startposition des SID ...` | ca. 1 s nicht bewegen |
| `Station bereit. Ablauf starten mit 'start'.` | Zustand `idle`, alles bereit |
| `ACHTUNG: SID BEWEGT SICH AUTOMATISCH` + Befehlsliste | ab jetzt nehmen Terminal und MQTT Befehle an |

Eine „Tippe START“-Abfrage gibt es nicht mehr. Der erste `start` startet direkt den Ablauf.

### Ablauf starten

Im Frontend **START** (F2) drücken (Ablauf vorher unten links wählen: Alles / Liefern / Holen),
oder im Terminal eintippen:

```
start
```

oder per MQTT: `mosquitto_pub -h 192.168.2.1 -t station/cmd -m start`.

Anhalten mit **PAUSE** (F3): der laufende Schritt wird noch fertig, dann steht der Ablauf;
**WEITER** (F3) setzt fort. Nach einem Fehler oder Not-Aus: Ursache beheben, dann
**QUITTIEREN** (F8).

Während des Ablaufs auf die **gelben ⏳-Meldungen** achten. Bei
`BAND: Bitte jetzt Stütze N auf das Fließband stellen (10 s)` die Stütze von der
Erkennungsfläche auf den Anfang des Bands stellen.

Am Ende: `FERTIG` und `Alle Stützen sind wieder an der Abgabe.` Die Station ist wieder
`idle`, der nächste `start` kann folgen.

### Beenden

1. Falls noch etwas läuft: `stop` und warten, bis die Station `idle` oder `docked` meldet.
2. **BEENDEN** (F10) im Frontend oder `quit` im Terminal. Das Programm stoppt alle Geräte,
   schließt das Frontend und beendet die Ortung, sofern es sie selbst gestartet hat.
3. SID hinsetzen und ausschalten, dann Band und Greifer.

---

## Teil D – Erste Inbetriebnahme / nach Umbauten: schrittweise testen

Nach Änderungen an Aufbau, Kalibrierung oder Code nicht direkt `start` ausführen, sondern
von unten nach oben testen:

| Schritt | Aufruf | Erwartung |
|---|---|---|
| 1. Band allein | `python scripts/conveyor_console.py` → `3` | Band fährt 3 s, Status `running` → `idle` |
| 2. Greifer allein | `python scripts/arm_console.py` → `home`, `max`, `min`, `open`, `close` | Achse fährt, Zange öffnet/schließt |
| 3. YOLO allein | `python scripts/F_stilt_erkennung_YOLO.py` → Leertaste | Stütze auf der Erkennungsfläche wird gefunden |
| 4. Ortung allein | `python scripts/aruco_tracker_calibrated.py` | Status `tracking`, Pfeil zeigt in Laufrichtung |
| 5. Ablauf ohne echten SID | `python main.py --sim-nav` → `start` | Band, Greifer und Kamera arbeiten, der SID wird nur simuliert |
| 6. Navigation einzeln | `python main.py` → `goto 2.0 0.0`, `dock`, `undock` | SID erreicht Ziele, dockt und fährt wieder aus |
| 7. Teilabläufe | `liefern`, danach `holen` | Stützen werden gestellt bzw. zurückgebracht |
| 8. Gesamtablauf | `start` | kompletter Ablauf |

Zwischen den Einzeltests (Schritte 1 bis 4) die Konsolen wieder schließen, bevor
`main.py` startet. Ausnahme: Die Ortung aus Schritt 4 darf weiterlaufen, `main.py`
erkennt sie.

---

## Teil E – Troubleshooting

### E1 Grundregel bei jedem Fehler

1. Die rote Meldung im Terminal lesen: Sie nennt den Schritt und den Grund.
2. Details stehen in der Logdatei `logs/station_<Datum>_<Zeit>.log` (Pfad steht beim Start
   im Terminal), z. B. mit `tail -100 logs/$(ls logs | tail -1)`.
3. Zustand prüfen: `status`.
4. Wieder bereit machen:

| Zustand / Lage des SID | Befehlsfolge |
|---|---|
| `error`, SID frei (x > 1,2 m) | `reset` |
| `error`, SID im Dock oder in der Zone | `undock`, danach ggf. `reset` |
| `estop`, SID frei oder im Dock | `reset` |
| `estop`, SID in der Zone, aber nicht im Dock | `quit`, Programm neu starten, dann `undock` und `reset` |
| `docked` und `start` soll folgen | `undock` |

Nach einem abgebrochenen Ablauf: Stützen im Greifer, auf dem Band oder auf den Plätzen von
Hand in den Ausgangszustand bringen (siehe B3), bevor neu gestartet wird.

### E2 Beim Programmstart

| Symptom | Ursache | Abhilfe |
|---|---|---|
| Frontend: System „MQTT-Broker“ rot (`192.168.2.1 nicht erreichbar`), Band/Greifer „kein Broker“ | Broker aus, falsches Netz | `ping 192.168.2.1`; Broker bzw. Router prüfen; Netzwerkverbindung des PCs prüfen. Das Programm läuft weiter und verbindet sich selbst, sobald der Broker da ist; die Startpose wird dann automatisch gemessen |
| Frontend zeigt `FAILED`, System „SID (Lite3)“: `Start fehlgeschlagen: …` | SID konnte nicht angelegt werden (Modelldatei in `models/` fehlt, UDP-Port belegt – z. B. weil das Bewegungstraining gerade sammelt) | Grund steht im Frontend; beheben und `main.py` neu starten |
| `FileNotFoundError: …/models/dynamics_model.joblib` | Modell fehlt in `models/` | Modell aus `models/archiv/` zurückkopieren oder neu trainieren (Bewegungstraining) |
| `ModuleNotFoundError` (`ultralytics`, `joblib`, `sklearn` …) | venv nicht aktiv oder Pakete fehlen | `source .venv/bin/activate`; Pakete aus Teil A nachinstallieren |
| `Fließband NICHT erreichbar` | ESP32 aus, nicht im WLAN, kein Status | Band-ESP32 neu starten; `mosquitto_sub -h 192.168.2.1 -t station/conveyor/status -v` muss `idle` zeigen |
| `Greifer NICHT erreichbar` | ESP32 aus, keine Heartbeats seit 10 s | Greifer-ESP32 neu starten; `mosquitto_sub ... -t 'station/arm/#' -v` prüfen |
| `YOLO-Erkennung nicht verfügbar: YOLO model could not be loaded` | `best.pt` fehlt oder `ultralytics` fehlt | `models/best.pt` prüfen; `pip install ultralytics` |
| `YOLO-Erkennung nicht verfügbar: No video node delivering YUYV frames` | IR-Kamera nicht eingesteckt oder von einem anderen Programm belegt | Kamera umstecken; andere Kameraprogramme schließen; `v4l2-ctl --list-devices` |
| YOLO benutzt die falsche Kamera | eine andere Kamera liefert ebenfalls YUYV | in `F_stilt_erkennung_YOLO.py` `NAME_HINT` auf einen Teil des Kameranamens setzen (`cat /sys/class/video4linux/video*/name`) |
| `SID-Ortung meldet nach 60 s kein 'tracking'` | Referenzmarker oder SID-Marker nicht sichtbar, Ortung abgestürzt | Ortungsbild im Frontend und „Systeme > SID-Ortung“ ansehen: Status `teaching` = Referenzmarker verdeckt, `lost` = SID-Marker nicht im Bild; Marker freilegen, Licht prüfen |
| Danach `Station im FEHLER-Zustand: Keine initiale Pose empfangen` | Startpose konnte nicht gemessen werden | Programm beenden, Ursache (Ortung) beheben, neu starten. `reset` allein hilft nicht, Fahrbefehle bleiben dann gesperrt („Nicht initialisiert“) |
| Station startet in `error` („Start in der Sicherheitszone“) | SID steht beim Start in der Zone | `undock`, danach `reset` |
| Station startet in `docked` | SID steht im Dock | vor `start` erst `undock` |
| Ortungsbild „KEIN BILD“, Log/Terminal meldet `Keine Kamera mit 'arducam' im Namen` | Arducam nicht erkannt | USB-Kabel prüfen; Geräteliste im Log ansehen |
| Ortung bricht ab: Seitenverhältnis passt nicht zur Kalibrierung | falsche Auflösung | ohne `--resolution` starten (Kalibrierung gilt für 1920×1080) |
| Ortung läuft nicht, `main.py` meldet aber „läuft bereits“ | ein alter, hängender Tracker-Prozess | `pkill -f aruco_tracker`, dann `main.py` neu starten |
| Frontend öffnet sich nicht (z. B. über SSH) | kein Bildschirm oder kein Browser | `python main.py --no-window`, Adresse aus dem Terminal im Browser öffnen; Not-Aus sonst über `estop` |
| Frontend: Port 8080 belegt | zweite Instanz oder anderes Programm | anderes `main.py` beenden oder `port` in `config/frontend.yaml` ändern |
| Frontend: Ortungsbild „KEIN BILD“ | Ortung läuft ohne `--preview` (z. B. von Hand gestartet) | `pkill -f aruco_tracker`, `main.py` neu starten (startet die Ortung mit Vorschau) |

### E3 Befehle werden abgelehnt

| Meldung | Bedeutung | Abhilfe |
|---|---|---|
| `'start' ist im Zustand 'error' nicht erlaubt` | Station steht im Fehler | siehe E1 (`reset` bzw. `undock`) |
| `'start' im Zustand 'idle' gerade nicht möglich` + `Befehl abgelehnt: … nicht erreichbar` | ein benötigtes Gerät fehlt | Gerät prüfen (E2). Für `holen` wird nur der Greifer gebraucht |
| `'reset' im Zustand 'error' gerade nicht möglich` | SID steht in der Zone, aber nicht im Dock | erst `undock` |
| `'reset' ist im Zustand 'idle' nicht erlaubt` | es gibt nichts zurückzusetzen | – |
| `Unbekannter Befehl: …` | Tippfehler | `help` |
| `Ziel … liegt in der Sicherheitszone` / `außerhalb des Sicherheitsradius` | `goto` auf unzulässiges Ziel | Ziel mit x ≥ 1,2 m und höchstens 4 m von der Startpose wählen |

### E4 Während der Fahrt (Navigation)

| Meldung | Ursache | Abhilfe |
|---|---|---|
| `Tracker liefert während der Bewegung keine aktuelle Pose!` (Sicherheitsstopp) | SID-Marker 1,5 s lang nicht gesehen: verdeckt (Stütze, Greifer, Kabel, Person), Spiegelung, Rand des Kamerabilds, Ortung überlastet | Ortungsbild im Frontend auf der gefahrenen Strecke beobachten; Marker so befestigen, dass Greifer und Stütze ihn nicht verdecken; Licht und Spiegelungen prüfen. Danach `undock` (falls in der Zone) bzw. `reset` |
| `Keine Posen im Messfenster! Tracker ausgefallen?` | wie oben, aber im Stand nach einem Schritt | wie oben; kommt das Ortungsbild im Frontend noch? |
| `Roboter steht in der Sicherheitszone (x=1.1… m)!` | SID ist beim Start eines Befehls schon in der Zone (z. B. nach Rutschen beim Ausdocken) | `undock`, notfalls den SID von Hand auf x > 1,3 m stellen und das Programm neu starten |
| `Dock-Vorposition nicht genau genug erreicht (… seitlich −4.7 cm …)` | seitliche Toleranz 4 cm / 4° an der Vorposition verfehlt | erneut versuchen (`reset`, `start`). Tritt es oft auf: Laufmodell nachtrainieren (DOKUMENTATION §12). `DOCK_ALLOW_RELAXED = True` nur im Notfall |
| `Docken NICHT genau genug: längs −12.7 cm (zu kurz)` | Geradeauslauf ins Dock zu kurz (Untergrund, Akku, Modell) | `undock`, `reset`. Tritt es wiederholt auf: in `navigate.py` `DOCK_DISTANCE_OFFSET` erhöhen (z. B. `0.05`) oder `DOCK_SPEED` / `DOCK_DEADTIME` fest eintragen. Die geschätzten Werte stehen in der Logdatei |
| `Winkelabweichung in der Andockbahn …` / `Seitliche Abweichung von der Andocklinie …` | SID ist beim Einfahren schief gelaufen (> 20° bzw. > 20 cm) | `undock`; Dock und Untergrund prüfen, Marker fest? |
| `Rückwärts führt nicht aus der Zone … bitte manuell eingreifen` | SID steht in der Zone falsch herum | SID von Hand aus der Zone holen, Programm neu starten |
| `Ausdocken nach 3 Läufen nicht abgeschlossen` | SID kommt rückwärts nicht über x = 1,3 m | Hindernis hinter dem SID? Erneut `undock` |
| `Nicht ganz genau angekommen (…) – geht trotzdem weiter.` | nur eine Warnung beim Anfahren von Platz oder Warteplatz | nichts tun. Wird der Fehler groß, Laufmodell nachtrainieren |
| `Bereits … cm über den Punkt hinaus – kein weiterer Vorwärtsschritt` | Endanflug an eine Stütze etwas zu weit | nur eine Warnung. Prüfen, ob die Stütze umgestoßen wurde |
| SID reagiert gar nicht auf Befehle | SID nicht im Netz, nicht im Stand, oder UDP-Port 12345 von einem anderen Skript belegt | `ping 192.168.1.120`; SID aufstehen lassen; andere Skripte (`navigate.py`, `collect_*.py`) beenden |
| SID läuft deutlich an den Zielen vorbei | Laufmodell passt nicht mehr (Untergrund, Firmware, neue Kalibrierung) | Daten neu sammeln und trainieren (DOKUMENTATION §12); mit `check_heading.py` die Winkelmessung prüfen |

### E5 Greifer

| Meldung | Ursache | Abhilfe |
|---|---|---|
| `… Greifer meldet einen Fehler (Details in der Logdatei)` | ESP32 hat einen Fehler gemeldet oder 20 s lang keine Fertigmeldung geschickt | In der Logdatei nach `ESP32:` suchen. Häufig: Achse blockiert, Endschalter, Stromversorgung. Mit `arm_console.py` → `state`, `home` testen |
| `Greifer: offline` / `connection_lost` → Zustand `error` | ESP32 hat die Verbindung verloren | ESP32-Stromversorgung und WLAN prüfen; nach Wiederkehr `reset` |
| `Ziel … Schritte ausserhalb 0..11000, nicht gesendet` | Befehl außerhalb der Grenzen (`arm_to` > 44 cm) | Wert korrigieren. Grenzen stehen in `arm_config.py` |
| `ESP32: limit_reached …` beim Runterfahren | normal: unten ist der Endschalter | nichts tun |
| Stütze rutscht aus der Zange | Teilesensor ist aus, das wird nicht bemerkt | Zange mechanisch prüfen; `GRIP_CLOSED_US` in `arm_config.py` anpassen und in `arm_console.py` mit `config` übernehmen |

### E6 Fließband und Erkennung

| Symptom | Ursache | Abhilfe |
|---|---|---|
| Station bleibt bei `KAMERA: Suche Stütze N …` stehen | keine Stütze auf der Erkennungsfläche, oder YOLO erkennt sie nicht | Stütze auf die Fläche stellen. Wird sie trotzdem nicht erkannt: Live-Bild „Kamera Greifarm“ im Frontend prüfen (Lage, Licht, Verschmutzung); `scripts/F_stilt_erkennung_YOLO.py` allein testen. Abbruch mit `stop` |
| `Kein Bild von der YOLO-Kamera.` | IR-Kamera abgezogen oder hängt | Kamera neu einstecken, Programm neu starten |
| `Das Fließband hat die Fahrt nicht beendet.` / `Fließband meldet nach 23 s kein Fahrtende` | ESP32 hat den Befehl abgelehnt, war nicht `idle` oder hat sich aufgehängt | `mosquitto_sub … -t 'station/conveyor/#' -v`; ESP32 neu starten |
| Stütze liegt nicht am Bandende, wenn der SID einfährt | Fahrzeit passt nicht | `CONVEYOR_RUN_S` in `main.py` anpassen (Standard 18 s) |
| 10 s reichen nicht zum Aufstellen | – | `HAND_PLACE_S` in `main.py` erhöhen |

### E7 MQTT und Netzwerk

| Symptom | Ursache | Abhilfe |
|---|---|---|
| Warnungen `Verbindung zum Broker verloren (Keep alive timeout), verbinde neu...` | Netzwerk kurz weg oder der PC war im Ruhezustand | Die Clients verbinden sich automatisch neu. Energiesparmodus des PCs abschalten. Häufen sich die Meldungen: WLAN/Netzwerk prüfen |
| Geräte trennen sich gegenseitig | zwei Clients mit derselben Client-ID (z. B. `arm_console.py` läuft doppelt) | doppelte Skripte beenden |
| `station/state` zeigt `offline` | `main.py` läuft nicht (Last Will) | Programm starten |

### E8 Wenn gar nichts mehr hilft

1. `estop` bzw. NOT-AUS/Leertaste im Frontend, danach `quit`.
2. SID von Hand an eine freie Stelle außerhalb der Zone stellen, Greifer leeren.
3. Band- und Greifer-ESP32 neu starten. Hängt die Ortung, `pkill -f aruco_tracker`.
4. `python main.py` neu starten und nach Teil C vorgehen.
5. Die Logdatei des fehlerhaften Laufs aufbewahren (`logs/`) und bei der Fehlersuche
   mitschicken.
