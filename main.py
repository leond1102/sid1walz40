"""
main.py – Zentrale Steuerung der Station.

Startet zuerst das Frontend (Browser-Vollbild, siehe scripts/station_frontend.py und
config/frontend.yaml), dann alle Geräte, den Zustandsautomaten (Bibliothek `transitions`)
und die MQTT-Schnittstelle. Fehlt ein Gerät oder der Broker, läuft alles andere weiter;
die Systemübersicht im Frontend zeigt, was läuft und was nicht. MQTT-Verbindungen werden
im Hintergrund aufgebaut und nach Abbrüchen selbst neu versucht.

Geräte (Module in scripts/):
    nav        SID (Lite3) laufen, docken, ausdocken   navigate.py
    conveyor   Fließband                                conveyor_controller.py
    arm        Greifer auf dem SID: Achse + Zange       arm_controller.py / arm_config.py
    detector   YOLO-Erkennung der Stützen               F_stilt_erkennung_YOLO.py

Ablauf (Befehl "start"), alle Positionen siehe ABLAUF unten:
    1. AUSLIEFERN    Greifer referenzieren und hoch. Für jede der 3 Stützen:
                     Kamera: alle 0,5 s suchen, bis YOLO eine Stütze auf der Erkennungsfläche sieht.
                     Band: 10 s warten (Stütze von Hand aufs Band), dann 18 s fahren.
                     SID:  gleichzeitig zur Dock-Vorposition, Greifer auf.
                           Liegt die Stütze am Bandende: rein ins Dock, Greifer zu, raus,
                           zum Platz, Greifer runter, auf, rückwärts weg, Greifer zu und hoch.
    2. PAUSE         SID wartet 30 s bei (1.5, 0).
    3. ZURÜCKHOLEN   Für jede Stütze (links, y = -0.6, zuerst): Vorposition, Greifer runter und auf,
                     gerade an die Stütze, Greifer zu und hoch, gerade in die Abgabe (Zone),
                     Greifer auf, gerade rückwärts wieder raus.
    "liefern" macht nur 1, "holen" nur 3.

Zustände:
    init        initialize() misst die Pose und wählt idle / docked / error
                (fehlt die Pose, wird automatisch neu gemessen, sobald die Ortung den SID sieht)
    idle        SID frei außerhalb der Sicherheitszone, wartet auf Befehle
    running     der Ablauf läuft (Fortschritt steht in phase / step, alle Schritte in plan)
    paused      Pause: der laufende Schritt wird fertig, dann hält der Ablauf an (resume = weiter)
    navigating / docking / undocking            Einzelbefehle Navigation
    conveying / detecting / arm_moving / gripping   Einzelbefehle Geräte
    docked      SID steht im Dock
    error       Fehler; steht der SID in der Zone, ist undock erlaubt, sonst reset
    estop       Not-Aus; nur reset

Befehle (MQTT auf station/cmd oder Tastatur):
    start | liefern | holen           Ablauf starten (nur aus idle)
    pause | resume                    Ablauf nach dem laufenden Schritt anhalten / weiterlaufen lassen
    stop                              laufenden Ablauf/Befehl sauber abbrechen
    estop                             alles sofort stoppen (auch NOT-AUS-Knopf / Leertaste im Frontend)
    reset                             error | estop -> idle | docked (fehlt die Startpose: wie init)
    init                              Startpose neu messen (nur in error, wenn sie beim Start fehlte)
    status                            aktuellen Zustand ausgeben
    Einzelbefehle zum Testen:
    goto x y [theta] | approach x y [winkel] [abstand] | dock | undock
    convey <s> | detect | home | arm_to <cm> | grip | release
    quit                              Programm beenden (Tastatur; im Frontend: Knopf BEENDEN / F10)

Nach jedem Zustandswechsel und jedem Schritt wird der Zustand als JSON auf
station/state veröffentlicht (retained).

Aufruf:
    python main.py              echter SID, startet die SID-Ortung falls nötig
    python main.py --sim-nav    SID simuliert, alle anderen Geräte echt
    python main.py --no-window  kein Fenster öffnen (Frontend trotzdem im Browser erreichbar)
    python main.py --cv-window  altes OpenCV-Kamerafenster statt Frontend
"""

import argparse
import json
import os
import select
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

# Die Gerätemodule liegen in scripts/ und importieren sich dort gegenseitig
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # keine OpenCV-Warnungen beim Kamerasuchen in der Konsole

import paho.mqtt.client as mqtt  # noqa: E402
from loguru import logger  # noqa: E402
from transitions import MachineError  # noqa: E402
from transitions.extensions import LockedMachine  # noqa: E402

import arm_config  # noqa: E402
from arm_controller import ArmController  # noqa: E402
from conveyor_controller import ConveyorRemote  # noqa: E402
from F_stilt_erkennung_YOLO import StiltDetector  # noqa: E402
from navigate import NavigationController, NavResult, NavStatus  # noqa: E402
from station_display import run_camera_window, say, setup_logging, start_tracker  # noqa: E402
from station_frontend import StationFrontend, SystemBoard  # noqa: E402

# --- MQTT ---
BROKER = "192.168.2.1"
PORT = 1883
CLIENT_ID = "station_main"
TOPIC_CMD = "station/cmd"                   # Befehle als Text, siehe oben
TOPIC_STATE = "station/state"               # Zustand als JSON (retained)
TOPIC_TRACKER_STATUS = "station/tracker/status"

# --- ABLAUF: Positionen in Kamerakoordinaten (m, Grad), gemeint ist immer die Position des SID ---
PLACE_POSITIONS = [(2.0, -0.6), (2.0, 0.0), (2.0, 0.6)]  # Reihenfolge beim Ablegen und Zurückholen
STILT_COUNT = len(PLACE_POSITIONS)
DETECT_RETRY_S = 0.5           # Pause zwischen zwei Suchversuchen der Kamera
PLACE_ANGLE = 0.0              # Blickrichtung an den Plätzen (0° = weg von der Station)
PLACE_STANDOFF = 0.5           # m: Vorposition so weit vor dem Platz
PAUSE_POSITION = (1.5, 0.0)    # hier wartet der SID, wenn alle Stützen stehen
PAUSE_S = 30
DROP_POSITION = (1.0, 0.25)    # Abgabe auf der Erkennungsfläche (in der Zone, Einfahrt wie beim Docken)
BACK_OFF_M = 0.30              # m: so weit geht der SID nach dem Ablegen rückwärts
GRIPPER_WAIT_S = 2.0           # nach jedem Öffnen/Schließen so lange warten, bis der Greifer ganz auf/zu ist
STEPPER_WAIT_S = 1.0           # nach jeder Fahrt der Greiferachse (home/min/max) so lange warten
HAND_PLACE_S = 10              # Zeit, um die Stütze von Hand aufs Band zu stellen (Roboterarm fehlt)
CONVEYOR_RUN_S = 18            # so lange fährt das Band, bis die Stütze am Ende ist
TRACKER_WAIT_S = 60            # so lange wird beim Start auf die SID-Ortung gewartet
INIT_RETRY_S = 15              # fehlt die Startpose, frühestens so oft neu messen (sobald die Ortung trackt)

# Meldungen des Greifers, die die Station in den Fehlerzustand versetzen
# (part_lost kommt nur, wenn arm_config.USE_PART_SENSOR = True ist)
ARM_FAULT_CODES = ("part_lost", "offline", "connection_lost")

NAV_STATES = ["navigating", "docking", "undocking"]
TASK_STATES = ["conveying", "detecting", "arm_moving", "gripping"]
CYCLE_STATES = ["running", "paused"]
BUSY_STATES = NAV_STATES + TASK_STATES + CYCLE_STATES
READY_STATES = ["idle", "docked"]  # hier dürfen Einzelbefehle der Geräte starten
CYCLE_MODES = ("alles", "liefern", "holen")


class CycleAborted(Exception):
    """Der Ablauf wurde mit stop / estop / fault abgebrochen."""


class StepFailed(Exception):
    """Ein Schritt des Ablaufs hat nicht geklappt."""


@dataclass
class Step:
    """Ein Schritt des Ablaufs. Die Liste aller Schritte steht vor dem Start fest (Anzeige: vorher/jetzt/als Nächstes)."""

    section: str                # AUSLIEFERN | PAUSE | ZURÜCKHOLEN
    stilt: int                  # Nummer der Stütze, 0 = gehört zu keiner Stütze
    text: str                   # Anzeige im Frontend
    run: Callable[[], None]     # führt den Schritt aus, wirft CycleAborted / StepFailed


class Station:
    """Zustandsautomat der Station. Befehle sind Methoden (Trigger) am Objekt."""

    STATES = ["init", "idle", "running", "paused", "navigating", "docking", "docked", "undocking",
              "conveying", "detecting", "arm_moving", "gripping", "error", "estop"]

    def __init__(self, nav: NavigationController, conveyor: ConveyorRemote,
                 arm: ArmController, detector: StiltDetector):
        self.nav = nav
        self.conveyor = conveyor
        self.arm = arm
        self.detector = detector

        self.message = ""                    # letzte Ergebnis-/Fehlermeldung
        self.phase = ""                      # Abschnitt des Ablaufs, z. B. "Ausliefern 2/3"
        self.step = ""                       # was der SID gerade tut
        self.band = ""                       # was das Fließband gerade tut
        self.detections: list[dict] = []     # Ergebnis der letzten Erkennung
        self.tracker_status = ""             # Status der SID-Ortung (tracking, lost, ...)
        self.prev_state = ""                 # Zustand vor dem letzten Wechsel
        self.plan: list[Step] = []           # Schritte des aktuellen bzw. letzten Ablaufs
        self.step_index = 0                  # Index des aktuellen Schritts in plan
        self.cycle_result = ""               # running | done | aborted | failed ("" = noch kein Ablauf)
        self.holding = False                 # True = Ablauf steht in der Pause still
        self._last_state = "init"
        self._abort = threading.Event()      # gesetzt = Ablauf / Einzelaktion abbrechen
        self._resume = threading.Event()     # nicht gesetzt = Pause, der Ablauf hält vor dem nächsten Schritt an
        self._resume.set()
        self._cycle_thread: threading.Thread | None = None
        self._feeding: Future | None = None  # Band-Strang der aktuellen Stütze
        self._band_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="band")
        self._mqtt: mqtt.Client | None = None

        self.arm.on_event = self._on_arm_event

        self.machine = LockedMachine(model=self, states=self.STATES, initial="init",
                                     after_state_change="_on_state_change", auto_transitions=False)
        t = self.machine.add_transition

        # Start: je nach gemessener Pose
        t("_started", "init", "docked", conditions="_is_docked")
        t("_started", "init", "error", conditions="_is_in_zone")   # in der Zone, aber nicht im Dock
        t("_started", "init", "idle")
        t("_start_failed", "init", "error")
        t("reinit", "error", "init", conditions="_not_initialized")   # Startpose fehlt: neu messen

        # Ablauf
        t("start", "idle", "running", conditions="_cycle_ready", after="_do_cycle")
        t("pause", "running", "paused", after="_do_pause")     # hält nach dem laufenden Schritt an
        t("resume", "paused", "running", after="_do_resume")
        for result in ("cycle_done", "cycle_aborted"):
            t(result, CYCLE_STATES, "docked", conditions="_is_docked")
            t(result, CYCLE_STATES, "idle", conditions="_is_free")
            t(result, CYCLE_STATES, "error")                        # SID steht noch in der Zone
        t("cycle_failed", CYCLE_STATES, "error")

        # Einzelbefehle Navigation
        t("goto", "idle", "navigating", after="_do_goto")
        t("approach", "idle", "navigating", after="_do_approach")
        t("dock", "idle", "docking", after="_do_dock")
        t("undock", ["docked", "error"], "undocking", conditions="_is_in_zone", after="_do_undock")

        # Ergebnisse der Navigation (aus dem Worker-Thread)
        t("nav_succeeded", "navigating", "idle")
        t("nav_succeeded", "docking", "docked")
        t("nav_succeeded", "undocking", "idle")
        t("nav_aborted", "navigating", "idle")
        t("nav_aborted", ["docking", "undocking"], "error")        # SID steht evtl. in der Andockbahn
        t("nav_rejected", ["navigating", "docking"], "idle")       # Befehl unzulässig, nichts bewegt
        t("nav_rejected", "undocking", "docked", conditions="_is_docked")
        t("nav_rejected", "undocking", "error")
        t("nav_failed", "docking", "docked", conditions="_is_docked")
        t("nav_failed", NAV_STATES, "error")

        # Einzelbefehle Geräte (nur wenn das Gerät erreichbar ist)
        t("convey", READY_STATES, "conveying", conditions="_conveyor_ready", after="_do_convey")
        t("detect", READY_STATES, "detecting", conditions="_detector_ready", after="_do_detect")
        t("home", READY_STATES, "arm_moving", conditions="_arm_ready", after="_do_home")
        t("arm_to", READY_STATES, "arm_moving", conditions="_arm_ready", after="_do_arm_to")
        t("grip", READY_STATES, "gripping", conditions="_arm_ready", after="_do_grip")
        t("release", READY_STATES, "gripping", conditions="_arm_ready", after="_do_release")

        # Ergebnisse der Einzelbefehle Geräte (aus dem Worker-Thread)
        t("task_done", TASK_STATES, "docked", conditions="_is_docked")
        t("task_done", TASK_STATES, "idle")
        t("task_aborted", TASK_STATES, "docked", conditions="_is_docked")
        t("task_aborted", TASK_STATES, "idle")
        t("task_failed", TASK_STATES, "error")

        # Abbruch, Fehler, Not-Aus
        t("stop", BUSY_STATES, None, after="_do_stop")               # Ergebnis kommt als *_aborted
        t("fault", READY_STATES + TASK_STATES + CYCLE_STATES, "error", after="_do_stop_devices")
        t("estop", "*", "estop", after="_do_estop")
        t("reset", ["error", "estop"], "docked", conditions="_is_docked")
        t("reset", ["error", "estop"], "idle", conditions="_is_free")

    # ==================================================================
    #  Start / Ende
    # ==================================================================
    def initialize(self) -> NavResult:
        """Misst die Startpose des SID und wählt den ersten Zustand."""
        say("step", "Messe die Startposition des SID ...")
        res = self.nav.initialize()
        self.message = res.message
        if res.ok:
            self._started()
        else:
            self._start_failed()
        return res

    def reinitialize(self) -> bool:
        """Misst die Startpose erneut, wenn sie beim Start nicht gemessen werden konnte (z. B. Ortung fehlte)."""
        try:
            if not self.trigger("reinit"):
                return False
        except MachineError:
            return False
        return self.initialize().ok

    def shutdown(self) -> None:
        """Bricht alles ab und stoppt die Geräte (beim Programmende)."""
        self._abort.set()
        self._band_pool.shutdown(wait=False, cancel_futures=True)

    def attach_mqtt(self, client: mqtt.Client) -> None:
        """Verbindet den Automaten mit dem Broker: Befehle empfangen, Zustand senden."""
        self._mqtt = client
        client.on_connect = self._on_mqtt_connect
        client.on_message = self._on_mqtt_message
        client.will_set(TOPIC_STATE, json.dumps({"state": "offline"}), qos=1, retain=True)

    # ==================================================================
    #  Bedingungen
    # ==================================================================
    def _not_initialized(self, *args, **kwargs):
        return not self.nav.initialized

    def _is_docked(self, *args, **kwargs):
        return self.nav.location == NavigationController.DOCKED

    def _is_in_zone(self, *args, **kwargs):
        return self.nav.location != NavigationController.FREE

    def _is_free(self, *args, **kwargs):
        return self.nav.location == NavigationController.FREE

    def _conveyor_ready(self, *args, **kwargs):
        return self._check(self.conveyor.is_connected(), "Fließband nicht erreichbar")

    def _arm_ready(self, *args, **kwargs):
        return self._check(self.arm.is_connected(), "Greifer nicht erreichbar")

    def _detector_ready(self, *args, **kwargs):
        return self._check(self.detector.is_connected(), "Kamera/YOLO nicht bereit")

    def _cycle_ready(self, mode="alles"):
        if mode == "holen":
            return self._arm_ready()
        return self._arm_ready() and self._conveyor_ready() and self._detector_ready()

    @staticmethod
    def _check(ok: bool, reason: str) -> bool:
        if not ok:
            say("error", f"Befehl abgelehnt: {reason}.")
        return ok

    # ==================================================================
    #  ABLAUF
    # ==================================================================
    def _do_cycle(self, mode="alles"):
        self._abort.clear()
        self._resume.set()
        self.holding = False
        self._feeding = None
        self.plan = self._build_plan(mode)
        self.step_index = 0
        self.cycle_result = "running"
        self._cycle_thread = threading.Thread(target=self._cycle_worker, daemon=True, name="ablauf")
        self._cycle_thread.start()

    def _do_pause(self):
        self._resume.clear()
        say("wait", "PAUSE: Der laufende Schritt wird noch fertig, danach hält der Ablauf an. Weiter mit 'resume'.")

    def _do_resume(self):
        self._resume.set()
        say("ok", "Ablauf läuft weiter.")

    def _cycle_worker(self) -> None:
        """Führt den Ablauf aus und meldet das Ergebnis an den Automaten."""
        try:
            self._run_plan()
        except CycleAborted:
            self.message = "Ablauf abgebrochen"
            say("error", "Ablauf wurde abgebrochen.")
            self.cycle_result = "aborted"
            trigger = "cycle_aborted"
        except StepFailed as e:
            self.message = str(e)
            say("error", f"FEHLER: {e}")
            self._abort.set()   # stoppt auch das Band im Hintergrund
            self._stop_devices()
            self.cycle_result = "failed"
            trigger = "cycle_failed"
        except Exception as e:  # damit der Automat nie in "running" hängen bleibt
            logger.exception("Unerwarteter Fehler im Ablauf")
            self.message = f"Unerwarteter Fehler: {e}"
            say("error", self.message)
            self._abort.set()
            self._stop_devices()
            self.cycle_result = "failed"
            trigger = "cycle_failed"
        else:
            self.message = "Ablauf fertig"
            say("phase", "FERTIG")
            say("ok", "Alle Stützen stehen auf ihren Plätzen." if self.plan[-1].section == "AUSLIEFERN"
                else "Alle Stützen sind wieder an der Abgabe. Neu starten mit 'start'.")
            self.cycle_result = "done"
            trigger = "cycle_done"
        self.holding = False
        self.phase = self.step = self.band = ""
        self._safe_trigger(trigger)

    def _run_plan(self) -> None:
        """Arbeitet die Schritte der Reihe nach ab. In der Pause wird vor dem nächsten Schritt gewartet."""
        section, stilt = "", 0
        for index, step in enumerate(self.plan):
            self.step_index = index
            self._check_abort()
            self._pause_gate()
            if step.section != section:
                section, stilt = step.section, 0
                say("phase", section)
            if step.stilt != stilt:
                stilt = step.stilt
                say("info", f"--- Stütze {stilt} von {STILT_COUNT} ---")
            self.phase = section.capitalize() + (f" {stilt}/{STILT_COUNT}" if stilt else "")
            self.publish_state()
            step.run()
        self.step_index = len(self.plan)

    def _build_plan(self, mode: str) -> list[Step]:
        """Stellt alle Schritte des Ablaufs zusammen (Positionen siehe ABLAUF oben)."""
        plan: list[Step] = []
        section, stilt = "", 0

        def add(text: str, run: Callable[[], None]) -> None:
            plan.append(Step(section, stilt, text, run))

        def nav(text: str, command, *args, exact: bool = True, then: Callable[[], None] | None = None) -> None:
            def run():
                self._nav_step(text, command, *args, exact=exact)
                if then is not None:
                    then()
            add(text, run)

        def arm(text: str, command, then: Callable[[], None] | None = None) -> None:
            def run():
                self._arm_step(text, command)
                if then is not None:
                    then()
            add(text, run)

        # ---------- 1. Ausliefern ----------
        if mode in ("alles", "liefern"):
            section = "AUSLIEFERN"
            arm("Greifer: Referenzfahrt", self.arm.home)
            arm("Greifer fährt hoch (max)", self.arm.move_max, then=partial(self._start_feeding, 1))

            for number, (x, y) in enumerate(PLACE_POSITIONS, start=1):
                stilt = number
                # Abholen am Bandende
                nav("SID läuft zur Vorposition vor der Dockingstation", self.nav.dock_pre)
                arm("Greifer öffnet sich für die Übergabe", self.arm.gripper_open)
                add(f"SID wartet, bis Stütze {number} am Bandende ist", partial(self._wait_feeding, number))
                nav("SID läuft in die Dockingstation", self.nav.dock)
                arm("Greifer schließt – SID übernimmt die Stütze", self.arm.gripper_close)
                # danach ist das Band frei für die nächste Stütze
                nav("SID verlässt die Dockingstation", self.nav.undock,
                    then=partial(self._start_feeding, number + 1) if number < STILT_COUNT else None)

                # Abstellen am Platz
                nav(f"SID trägt Stütze {number} zum Platz ({x}, {y})",
                    self.nav.approach, x, y, PLACE_ANGLE, PLACE_STANDOFF, exact=False)
                arm("Greifer fährt runter (min)", self.arm.move_min)
                arm("Greifer öffnet – Stütze steht", self.arm.gripper_open)
                nav("SID geht rückwärts von der Stütze weg", self.nav.back_off, BACK_OFF_M)
                arm("Greifer schließt", self.arm.gripper_close)
                arm("Greifer fährt hoch (max)", self.arm.move_max,
                    then=partial(say, "ok", f"Stütze {number} steht auf ({x}, {y})."))

        # ---------- 2. Pause ----------
        if mode == "alles":
            section, stilt = "PAUSE", 0
            nav(f"SID läuft zum Warteplatz {PAUSE_POSITION}", self.nav.go_to, *PAUSE_POSITION, None, exact=False)
            text = f"SID wartet {PAUSE_S} Sekunden"
            add(text, partial(self._wait, text, PAUSE_S))

        # ---------- 3. Zurückholen ----------
        if mode in ("alles", "holen"):
            section, stilt = "ZURÜCKHOLEN", 0
            add("Greifer: Referenzfahrt (falls nötig)", self._home_if_needed)

            for number, (x, y) in enumerate(PLACE_POSITIONS, start=1):
                stilt = number
                nav(f"SID läuft zur Vorposition von Stütze {number}",
                    self.nav.approach_pre, x, y, PLACE_ANGLE, PLACE_STANDOFF, exact=False)
                arm("Greifer fährt runter (min)", self.arm.move_min)
                arm("Greifer öffnet", self.arm.gripper_open)
                nav("SID läuft gerade an die Stütze", self.nav.approach_final, x, y, PLACE_ANGLE, exact=False)
                arm("Greifer schließt – Stütze aufgenommen", self.arm.gripper_close)
                arm("Greifer fährt hoch (max)", self.arm.move_max)

                nav(f"SID bringt Stütze {number} zur Abgabe {DROP_POSITION}", self.nav.enter_zone, *DROP_POSITION)
                arm("Greifer öffnet – Stütze abgegeben", self.arm.gripper_open)
                nav("SID geht rückwärts aus der Abgabe", self.nav.undock,
                    then=partial(say, "ok", f"Stütze {number} ist zurück an der Abgabe."))
        return plan

    # ---------- Band-Strang (läuft parallel zum SID) ----------
    def _start_feeding(self, number: int) -> None:
        """Startet im Hintergrund: warten, bis die Stütze auf dem Band steht, dann Band fahren."""
        self._feeding = self._band_pool.submit(self._feed_stilt, number)

    def _find_stilt(self, number: int) -> None:
        """Sucht alle DETECT_RETRY_S Sekunden, bis YOLO mindestens eine Stütze sieht."""
        self.band = f"Kamera sucht Stütze {number}"
        say("wait", f"KAMERA: Suche Stütze {number} auf der Erkennungsfläche (alle {DETECT_RETRY_S} s) ...")
        self.publish_state()
        while True:
            detections = self.detector.detect()
            if detections is None:
                raise StepFailed("Kein Bild von der YOLO-Kamera.")
            if detections:
                self.detections = detections
                say("ok", f"KAMERA: Stütze erkannt ({len(detections)} im Bild).")
                return
            self._sleep(DETECT_RETRY_S)

    def _feed_stilt(self, number: int) -> None:
        self._find_stilt(number)
        self.band = f"Stütze {number} von Hand aufs Band stellen"
        say("wait", f"BAND: Bitte jetzt Stütze {number} auf das Fließband stellen ({HAND_PLACE_S} s).")
        self._sleep(HAND_PLACE_S)
        self._pause_gate()   # in der Pause startet das Band nicht
        self.band = f"fährt Stütze {number} zur Dockingstation"
        say("step", f"BAND: Fließband fährt Stütze {number} zur Dockingstation ({CONVEYOR_RUN_S} s).")
        self.publish_state()
        ok = self.conveyor.drive(CONVEYOR_RUN_S, wait=True)
        self._check_abort()
        if not ok:
            raise StepFailed("Das Fließband hat die Fahrt nicht beendet.")
        self.band = f"Stütze {number} liegt bereit"
        say("ok", f"BAND: Stütze {number} liegt am Bandende bereit.")
        self.publish_state()

    def _wait_feeding(self, number: int) -> None:
        """Wartet, bis das Band die Stütze gebracht hat (Fehler des Bands werden hier weitergereicht)."""
        feeding = self._feeding
        if feeding is None:
            return
        if not feeding.done():
            self._do(f"SID wartet, bis Stütze {number} am Bandende ist", kind="wait")
        while True:
            try:
                feeding.result(timeout=0.2)
                return
            except FutureTimeout:
                self._check_abort()

    # ---------- Bausteine des Ablaufs ----------
    def _home_if_needed(self) -> None:
        if not self.arm.state.get("homed"):
            self._arm_step("Greifer: Referenzfahrt", self.arm.home)

    def _wait(self, text: str, seconds: float) -> None:
        self._do(text, kind="wait")
        self._sleep(seconds)

    def _do(self, text: str, kind: str = "step") -> None:
        """Meldet den nächsten Schritt (Konsole, Frontend, MQTT)."""
        self._check_abort()
        self.step = text
        say(kind, text)
        self.publish_state()

    def _nav_step(self, text: str, command, *args, exact: bool = True) -> None:
        """Navigationsbefehl ausführen. exact=False: ungenau angekommen ist nur eine Warnung."""
        self._do(text)
        res = command(*args)
        self._check_abort()
        if res.ok:
            return
        if res.status == NavStatus.ABORTED:
            raise CycleAborted()
        if not exact and res.status == NavStatus.FAILED:
            say("info", f"Nicht ganz genau angekommen ({res.message}) – geht trotzdem weiter.")
            return
        raise StepFailed(f"{text}: {res.message}")

    def _arm_step(self, text: str, command) -> None:
        """Greiferbefehl senden, warten bis der ESP32 fertig meldet, dann noch kurz Pause.

        Die Pause (GRIPPER_WAIT_S bzw. STEPPER_WAIT_S) gibt der Mechanik Zeit, wirklich
        zur Ruhe zu kommen, bevor der SID weiterläuft oder der nächste Befehl kommt.
        """
        self._do(text)
        ok = command(wait=True)
        self._check_abort()
        if not ok:
            raise StepFailed(f"{text}: Greifer meldet einen Fehler (Details in der Logdatei).")
        is_gripper = command.__name__.startswith("gripper")
        pause = GRIPPER_WAIT_S if is_gripper else STEPPER_WAIT_S
        if pause > 0:
            say("wait", f"{'Greifer' if is_gripper else 'Achse'}: {pause:g} s Pause")
            self._sleep(pause)

    def _sleep(self, seconds: float) -> None:
        """Wartet, kann aber jederzeit mit stop / estop abgebrochen werden. In der Pause steht die Uhr."""
        remaining = seconds
        while remaining > 0:
            self._pause_gate()
            started = time.monotonic()
            if self._abort.wait(min(remaining, 0.2)):
                raise CycleAborted()
            remaining -= time.monotonic() - started

    def _pause_gate(self) -> None:
        """Wartet, solange die Pause aktiv ist (stop / estop beenden das Warten)."""
        if self._resume.is_set():
            return
        in_cycle = threading.current_thread() is self._cycle_thread
        if in_cycle:
            self.holding = True
            say("wait", "Ablauf steht in der PAUSE.")
            self.publish_state()
        try:
            while not self._resume.wait(0.2):
                self._check_abort()
        finally:
            if in_cycle:
                self.holding = False
                self.publish_state()

    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise CycleAborted()

    # ==================================================================
    #  Einzelbefehle Navigation
    # ==================================================================
    def _do_goto(self, x, y, theta=None):
        self.nav.run_async(self.nav.go_to, x, y, theta, callback=self._on_nav_result)

    def _do_approach(self, x, y, angle=0.0, standoff=0.5):
        self.nav.run_async(self.nav.approach, x, y, angle, standoff, callback=self._on_nav_result)

    def _do_dock(self):
        self.nav.run_async(self.nav.dock, callback=self._on_nav_result)

    def _do_undock(self):
        self.nav.run_async(self.nav.undock, callback=self._on_nav_result)

    def _on_nav_result(self, res: NavResult):
        self.message = res.message
        trigger = {NavStatus.SUCCESS: "nav_succeeded", NavStatus.ABORTED: "nav_aborted",
                   NavStatus.REJECTED: "nav_rejected"}.get(res.status, "nav_failed")
        self._safe_trigger(trigger)

    # ==================================================================
    #  Einzelbefehle Geräte
    # ==================================================================
    def _do_convey(self, seconds):
        self._run_task(f"Fließband {seconds} s", lambda: self.conveyor.drive(seconds, wait=True))

    def _do_detect(self):
        def job():
            detections = self.detector.detect()
            if detections is None:
                return False
            self.detections = detections
            return True
        self._run_task("Erkennung", job)

    def _do_home(self):
        self._run_task("Referenzfahrt", lambda: self.arm.home(wait=True))

    def _do_arm_to(self, cm):
        self._run_task(f"Achse auf {cm} cm", lambda: self.arm.move_cm(cm, wait=True))

    def _do_grip(self):
        self._run_task("Greifen", lambda: self.arm.gripper_close(wait=True) and self.arm.has_part)

    def _do_release(self):
        self._run_task("Loslassen", lambda: self.arm.gripper_open(wait=True))

    def _run_task(self, name: str, job) -> None:
        """Führt eine blockierende Geräte-Aktion im eigenen Thread aus.

        job() gibt True bei Erfolg zurück. Das Ergebnis löst task_done,
        task_failed oder (nach stop) task_aborted aus.
        """
        self._abort.clear()

        def worker():
            try:
                ok = job()
            except Exception as e:  # damit der Automat nie in einem Zustand hängen bleibt
                logger.exception(f"{name}: unerwarteter Fehler")
                ok = False
                self.message = f"{name}: {e}"
            else:
                self.message = f"{name}: {'OK' if ok else 'fehlgeschlagen'}"
            if self._abort.is_set():
                self.message = f"{name}: abgebrochen"
                self._safe_trigger("task_aborted")
            else:
                self._safe_trigger("task_done" if ok else "task_failed")

        threading.Thread(target=worker, daemon=True, name=f"task-{name}").start()

    # ==================================================================
    #  Abbruch / Not-Aus
    # ==================================================================
    def _do_stop(self):
        say("error", "STOP: laufende Aktion wird abgebrochen.")
        self._abort.set()
        self.nav.abort()
        self._stop_devices()

    def _do_stop_devices(self, *args, **kwargs):
        self._abort.set()
        self.nav.abort()
        self._stop_devices()

    def _stop_devices(self):
        if self.conveyor.state == "running":
            self.conveyor.stop()
        self.arm.stop()

    def _do_estop(self):
        self._abort.set()
        self.nav.emergency_stop()
        self.conveyor.stop()
        self.arm.stop()
        say("error", "NOT-AUS ausgelöst! Weiter mit 'reset'.")

    # ==================================================================
    #  Meldungen der Geräte
    # ==================================================================
    def _on_arm_event(self, level: str, code: str, info: str) -> None:
        if code in ARM_FAULT_CODES:
            self.message = f"Greifer: {code} {info}".strip()
            say("error", self.message)
            self._safe_trigger("fault")

    def _safe_trigger(self, trigger: str) -> None:
        """Löst einen Übergang aus; passt er nicht zum Zustand (z. B. nach estop), wird er ignoriert."""
        try:
            self.trigger(trigger)
        except MachineError:
            logger.info(f"'{trigger}' im Zustand '{self.state}' ignoriert.")

    # ==================================================================
    #  Zustand melden
    # ==================================================================
    def _on_state_change(self, *args, **kwargs):
        self.prev_state, self._last_state = self._last_state, self.state
        texts = {
            "idle": ("ok", "Station bereit. Ablauf starten mit 'start'."),
            "docked": ("ok", "SID steht im Dock."),
            "error": ("error", f"Station im FEHLER-Zustand: {self.message.rstrip('.!')}. " + (
                "Weiter mit 'reset' (steht der SID in der Zone: erst 'undock')." if self.nav.initialized else
                "Die Startpose wird neu gemessen, sobald die Ortung den SID sieht (oder 'init').")),
            "estop": ("error", "Station im NOT-AUS. Weiter mit 'reset'."),
        }
        if self.state not in CYCLE_STATES + ["init"]:  # Ablauf und initialize() melden sich selbst
            kind, text = texts.get(self.state, ("info", f"Zustand: {self.state}"))
            say(kind, text)
        logger.info(f"[Station] Zustand -> {self.state.upper()} ({self.message})")
        self.publish_state()

    def status(self) -> dict:
        return {
            "state": self.state,
            "phase": self.phase,
            "step": self.step,
            "band": self.band,
            "message": self.message,
            "location": self.nav.location,
            "pose": self.nav.pose,
            "conveyor": self.conveyor.state,
            "arm": self.arm.state.get("state"),
            "gripper": self.arm.state.get("gripper"),
            "detections": len(self.detections),
            "tracker": self.tracker_status,
            "prev_state": self.prev_state,
            "cycle": self.cycle_result,
            "holding": self.holding,
            "step_index": self.step_index,
            "step_total": len(self.plan),
            "prev_step": self._plan_text(self.step_index - 1),
            "next_step": self._plan_text(self.step_index + 1),
            "devices_ok": {
                "conveyor": self.conveyor.is_connected(),
                "arm": self.arm.is_connected(),
                "detector": self.detector.is_connected(),
            },
            "time": time.time(),
        }

    def _plan_text(self, index: int) -> str:
        return self.plan[index].text if 0 <= index < len(self.plan) else ""

    def status_lines(self) -> list[str]:
        """Kurze Textzeilen für das Kamerafenster."""
        lines = [f"Station: {self.state.upper()}" + (f" | {self.phase}" if self.phase else "")]
        if self.step:
            lines.append(f"SID: {self.step}")
        if self.band:
            lines.append(f"Band: {self.band}")
        lines.append(f"Stuetzen erkannt: {len(self.detector.last_detections)}   [LEERTASTE = NOT-AUS]")
        return lines

    def publish_state(self) -> None:
        if self._mqtt is not None and self._mqtt.is_connected():
            self._mqtt.publish(TOPIC_STATE, json.dumps(self.status(), default=str), qos=1, retain=True)

    # ==================================================================
    #  Befehle (MQTT und Tastatur)
    # ==================================================================
    def handle_command(self, line: str) -> None:
        """Wertet einen Befehl als Text aus, z. B. "start", "goto 2.0 -0.5" oder "convey 5"."""
        parts = line.replace(",", " ").split()
        if not parts:
            return
        cmd, words = parts[0].lower(), parts[1:]

        if cmd == "status":
            say("info", str(self.status()))
            return
        if cmd == "init" or (cmd == "reset" and self.state == "error" and not self.nav.initialized):
            # ohne Startpose hilft reset nicht (Fahrbefehle bleiben gesperrt) -> neu messen
            if self.state == "error" and not self.nav.initialized:
                self.reinitialize()
            else:
                say("error", "'init' geht nur im Zustand error, wenn die Startpose beim Start fehlte.")
            return
        if cmd == "stop" and self.state not in BUSY_STATES:
            say("info", "Nichts zu stoppen.")
            return
        if cmd in ("start", "liefern", "holen"):
            mode = cmd if cmd != "start" else (words[0].lower() if words else "alles")
            if mode not in CYCLE_MODES:
                say("error", f"Unbekannter Ablauf '{mode}' (möglich: {', '.join(CYCLE_MODES)}).")
                return
            self._fire("start", mode)
            return

        try:
            args = [float(v) for v in words]
        except ValueError:
            say("error", f"Ungültige Zahl in '{line}'.")
            return

        # Befehl -> erlaubte Anzahl Zahlenwerte
        arg_counts = {
            "goto": (2, 3), "approach": (2, 3, 4), "dock": (0,), "undock": (0,),
            "convey": (1,), "detect": (0,), "home": (0,), "arm_to": (1,),
            "grip": (0,), "release": (0,), "stop": (0,), "estop": (0,), "reset": (0,),
            "pause": (0,), "resume": (0,),
        }
        if cmd not in arg_counts:
            say("error", f"Unbekannter Befehl: {cmd}")
            return
        if len(args) not in arg_counts[cmd]:
            say("error", f"'{cmd}' erwartet {' oder '.join(map(str, arg_counts[cmd]))} Zahlenwert(e).")
            return
        self._fire(cmd, *args)

    def _fire(self, trigger: str, *args) -> None:
        try:
            if not self.trigger(trigger, *args):
                say("error", f"'{trigger}' im Zustand '{self.state}' gerade nicht möglich.")
        except MachineError:
            say("error", f"'{trigger}' ist im Zustand '{self.state}' nicht erlaubt.")

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"Broker hat die Verbindung abgelehnt: {reason_code}")
            return
        logger.info(f"Station: mit dem MQTT-Broker verbunden, Befehle auf {TOPIC_CMD}.")
        client.subscribe([(TOPIC_CMD, 1), (TOPIC_TRACKER_STATUS, 1)])
        self.publish_state()

    def _on_mqtt_message(self, client, userdata, msg) -> None:
        text = msg.payload.decode(errors="replace").strip()
        if msg.topic == TOPIC_TRACKER_STATUS:
            self.tracker_status = text
            return
        say("info", f"Befehl per MQTT: {text}")
        self.handle_command(text)


HELP = ("Befehle: start | liefern | holen | pause | resume | stop | estop | reset | init | status | quit\n"
        "         Einzelbefehle siehe main.py; NOT-AUS im Frontend: Leertaste (einstellbar)")


def keyboard_loop(station: Station | None, quit_event: threading.Event) -> None:
    """Liest Befehle von der Tastatur, bis 'quit' eingegeben oder anders beendet wird.

    Liest nur, wenn wirklich eine Zeile da ist (select mit Timeout). So hängt der Thread beim
    Beenden über das Frontend nicht in input() fest (das bricht sonst Python beim Beenden ab).
    """
    try:
        while not quit_event.is_set():
            if not select.select([sys.stdin], [], [], 0.3)[0]:
                continue
            raw = sys.stdin.readline()
            if not raw:  # Dateiende
                # Kein Terminal (z. B. Start ohne Konsole): weiterlaufen, beendet wird über das Frontend
                if not sys.stdin.isatty():
                    return
                break
            line = raw.strip()
            if line.lower() in ("q", "quit", "exit"):
                break
            if line.lower() == "help":
                print(HELP)
            elif station is None:
                say("error", "Die Station läuft nicht (siehe Systeme im Frontend) – nur 'quit' möglich.")
            elif line:
                station.handle_command(line)
    except KeyboardInterrupt:
        pass
    quit_event.set()


def wait_for_tracker(station: Station, quit_event: threading.Event) -> bool:
    """Wartet bis zu TRACKER_WAIT_S, bis die SID-Ortung den SID verfolgt."""
    say("wait", "Warte auf die SID-Ortung ...")
    t_end = time.time() + TRACKER_WAIT_S
    while time.time() < t_end:
        if station.tracker_status == "tracking":
            say("ok", "SID-Ortung verfolgt den SID.")
            return True
        if quit_event.wait(0.5):
            return False
    say("error", f"SID-Ortung meldet nach {TRACKER_WAIT_S} s kein 'tracking' "
                 f"(Status: {station.tracker_status or 'unbekannt'}). Startpose wird neu gemessen, "
                 f"sobald die Ortung den SID sieht.")
    return False


def initialize_station(station: Station, quit_event: threading.Event, sim: bool) -> None:
    """Misst die Startpose. Klappt das nicht (Broker/Ortung fehlt), wird es wiederholt,
    sobald die Ortung den SID verfolgt. Läuft als Thread, damit Frontend und Tastatur sofort gehen."""
    if not sim:
        wait_for_tracker(station, quit_event)
    if quit_event.is_set():
        return
    station.initialize()
    if not sim:
        print("\n" + "!" * 50 + "\n! ACHTUNG: SID BEWEGT SICH AUTOMATISCH !\n" + "!" * 50)
    last_try = time.monotonic()
    while not quit_event.wait(1.0):
        if station.state != "error" or station.nav.initialized:
            continue
        if (not sim and station.tracker_status != "tracking") or time.monotonic() - last_try < INIT_RETRY_S:
            continue
        last_try = time.monotonic()
        say("step", "SID-Ortung ist da – messe die Startposition erneut ...")
        station.reinitialize()


def register_systems(systems: SystemBoard, client: mqtt.Client, sim_nav: bool, nav, station, conveyor, arm,
                     detector, tracker, startup: dict) -> None:
    """Meldet alle Teilsysteme mit ihrer Prüffunktion in der Systemübersicht an."""

    def broker():
        return ("ok", f"{BROKER} verbunden") if client.is_connected() else ("bad", f"{BROKER} nicht erreichbar")

    def tracking():
        if tracker is not None and tracker.poll() is not None:
            return "bad", f"Prozess beendet (Code {tracker.returncode})"
        status = station.tracker_status if station is not None else ""
        if not status and not client.is_connected():
            # Status und Pose kommen nur über den Broker (das Kamerabild auch lokal)
            return "bad", f"keine Daten – MQTT-Broker {BROKER} nicht erreichbar"
        return {
            "tracking": ("ok", "verfolgt den SID"),
            "lost": ("warn", "SID nicht im Bild"),
            "teaching": ("warn", "lernt die Referenzmarker ein"),
            "offline": ("bad", "offline"),
        }.get(status, ("bad", "kein Status – läuft die Ortung / der Broker?"))

    def sid():
        if nav is None:
            return "bad", f"Start fehlgeschlagen: {startup.get('nav', '?')}"
        if not sim_nav:
            latest = nav.robot.tracker.latest()
            if latest is None:
                if not nav.robot.tracker.connected:
                    return "bad", f"keine Pose – MQTT-Broker {BROKER} nicht erreichbar"
                return "bad", "keine Pose von der Ortung"
            age = time.time() - latest[0]
            if age > 1.0:
                return "warn", f"letzte Pose vor {age:.0f} s"
        if not nav.initialized:
            return "warn", "Startpose noch nicht gemessen"
        x, y, theta = nav.pose
        return "ok", f"{'Simulation · ' if sim_nav else ''}{nav.location} · x {x:.2f} y {y:.2f} θ {theta:.0f}°"

    def band():
        if conveyor.is_connected():
            return "ok", f"bereit ({conveyor.state})"
        return "bad", "kein Broker" if not client.is_connected() else f"meldet sich nicht (Status: {conveyor.state or '–'})"

    def gripper():
        if arm.is_connected():
            st = arm.state
            homed = "referenziert" if st.get("homed") else "nicht referenziert"
            return "ok", f"{st.get('state')} · Greifer {st.get('gripper')} · {homed}"
        return "bad", "kein Broker" if not client.is_connected() else "kein Lebenszeichen vom ESP32"

    def yolo():
        if detector.is_connected():
            return "ok", f"bereit (/dev/video{detector.camera_index})"
        if "detector" in startup:
            return "bad", startup["detector"]
        return "off", "wird gestartet ..."

    systems.set("MQTT-Broker", broker)
    systems.set("SID-Ortung", tracking if not sim_nav else ("off", "nicht nötig (Simulation)"))
    systems.set("SID (Lite3)", sid)
    systems.set("Fließband", band)
    systems.set("Greifer", gripper)
    systems.set("YOLO-Kamera", yolo)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zentrale Steuerung der Station")
    ap.add_argument("--sim-nav", action="store_true", help="SID simulieren (übrige Geräte echt)")
    ap.add_argument("--no-window", action="store_true",
                    help="kein Fenster öffnen (Frontend bleibt im Browser erreichbar, siehe config/frontend.yaml)")
    ap.add_argument("--cv-window", action="store_true",
                    help="altes OpenCV-Kamerafenster statt Frontend (Ortung im eigenen Fenster)")
    args = ap.parse_args()
    log_file = setup_logging(ROOT / "logs")

    say("phase", "STATION STARTET")
    say("info", f"Alle Details stehen in {log_file.relative_to(ROOT)}")

    # Zuerst die Anzeige: Sie läuft weiter, auch wenn danach ein Gerät oder die ganze Station nicht startet.
    quit_event = threading.Event()
    # Fenster zu (SIGHUP) oder kill (SIGTERM): sauber beenden, d. h. alle Geräte stoppen
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())
    signal.signal(signal.SIGHUP, lambda *_: quit_event.set())
    systems = SystemBoard()
    systems.start(quit_event)
    for name in ("MQTT-Broker", "SID-Ortung", "SID (Lite3)", "Fließband", "Greifer", "YOLO-Kamera"):
        systems.set(name, ("off", "wird gestartet ..."))
    frontend = None
    if not args.cv_window:
        try:
            frontend = StationFrontend(ROOT / "config", BROKER, PORT, systems, Station.STATES)
            frontend.on_quit = quit_event.set
            frontend.start()
            if not args.no_window:
                frontend.open_window()
        except (OSError, ValueError) as e:
            say("error", f"Frontend nicht verfügbar: {e}. Bedienung über die Tastatur.")
            if frontend is not None:
                frontend.stop()
            frontend = None

    nav = station = tracker = keyboard = None
    startup: dict[str, str] = {}   # Fehlertexte beim Start, für die Systemübersicht
    conveyor = ConveyorRemote(client_id="station_conveyor")
    arm = ArmController(client_id="station_arm")
    detector = StiltDetector(client_id="station_vision")
    detector.keep_empty = False   # Dauersuche: leere Bilder nicht speichern
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    try:
        if not arm_config.USE_PART_SENSOR:
            say("info", "Teilesensor am Greifer ist aus – Greifen gilt ohne Prüfung als erfolgreich.")
        say("step", "Lade die Laufmodelle des SID ...")
        try:
            nav = NavigationController.create(sim=args.sim_nav)
        except Exception as e:  # Modelle fehlen, UDP-Port belegt, ...
            logger.opt(exception=e).debug("SID konnte nicht angelegt werden")   # Traceback nur in die Logdatei
            startup["nav"] = f"{type(e).__name__}: {e}"
            say("error", f"SID konnte nicht gestartet werden ({startup['nav']}). "
                         f"Ohne SID läuft die Station nicht – Ursache beheben und neu starten.")
        if nav is not None:
            station = Station(nav, conveyor, arm, detector)
            station.attach_mqtt(client)
        elif frontend is not None:
            frontend.problem = "Station nicht gestartet: SID fehlt (siehe Systeme)"

        # Alle MQTT-Verbindungen im Hintergrund: Ohne Broker startet alles trotzdem und verbindet sich später.
        say("step", f"Verbinde mit dem MQTT-Broker {BROKER} (im Hintergrund) ...")
        conveyor.connect(background=True)
        arm.connect(background=True)
        client.connect_async(BROKER, PORT, keepalive=30)
        client.loop_start()

        if not args.sim_nav:
            tracker = start_tracker(window=frontend is None, preview=frontend is not None)
        register_systems(systems, client, args.sim_nav, nav, station, conveyor, arm, detector, tracker, startup)
        if frontend is not None:
            frontend.attach(station, detector)

        say("step", "Starte YOLO-Kamera und Modell ...")
        try:
            detector.connect()
            say("ok", f"YOLO-Kamera bereit (/dev/video{detector.camera_index}).")
        except RuntimeError as e:
            startup["detector"] = str(e)
            say("error", f"YOLO-Erkennung nicht verfügbar: {e}")

        if station is not None:
            threading.Thread(target=initialize_station, args=(station, quit_event, args.sim_nav),
                             daemon=True, name="start").start()
    except Exception as e:  # die Anzeige soll bleiben, auch wenn beim Start etwas Unerwartetes passiert
        logger.opt(exception=e).debug("Fehler beim Start")
        say("error", f"Fehler beim Start: {type(e).__name__}: {e}")
        if frontend is not None and station is None:
            frontend.problem = f"Start abgebrochen: {e}"

    try:
        print(HELP)
        keyboard = threading.Thread(target=keyboard_loop, args=(station, quit_event), daemon=True, name="tastatur")
        keyboard.start()
        if args.cv_window and not args.no_window and station is not None:
            run_camera_window(detector, station.status_lines, quit_event.is_set,
                              on_emergency=lambda: station.handle_command("estop"))
        else:
            if frontend is not None:
                say("info", f"Frontend: {frontend.url}  (Fenster zu? Einfach diese Adresse im Browser öffnen.)")
            quit_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        say("step", "Programm endet – stoppe alle Geräte ...")
        quit_event.set()
        if keyboard is not None:
            keyboard.join(timeout=1.0)   # gibt stdin frei, bevor Python beendet wird
        if station is not None:
            station.shutdown()
        if nav is not None:
            nav.shutdown()
        conveyor.disconnect()
        arm.stop()
        arm.disconnect()
        detector.disconnect()
        if client.is_connected():
            client.publish(TOPIC_STATE, json.dumps({"state": "offline"}), qos=1, retain=True).wait_for_publish(1.0)
        client.disconnect()
        client.loop_stop()
        if tracker is not None:
            tracker.terminate()
        if frontend is not None:
            frontend.stop()
        say("ok", "Beendet.")


if __name__ == "__main__":
    main()
