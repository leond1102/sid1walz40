"""
station_statemachine.py
Zustandsautomat der Roboterstation (Grundgerüst) mit der Bibliothek `transitions`.
    pip install transitions

Zustände:
    init        -> start() misst die Pose und wählt idle / docked / error
    idle        Roboter frei außerhalb der Sicherheitszone, wartet auf Befehle
    navigating  go_to / approach läuft
    docking     dock() läuft
    docked      Roboter steht im Dock (nur undock erlaubt)
    undocking   undock() läuft
    error       Fehler / Sicherheitsstopp; steht er in der Zone, ist undock erlaubt, sonst reset
    estop       Not-Aus; nur reset

Übergänge (Trigger = Methoden am Objekt):
    goto(x, y, theta=None)                idle -> navigating
    approach(x, y, angle=0, standoff=.5)  idle -> navigating
    dock()                                idle -> docking
    undock()                              docked | error (in Zone) -> undocking
    stop()                                laufenden Befehl sauber abbrechen
    estop()                               * -> estop (sofort stoppen)
    reset()                               error | estop -> idle | docked

Die Navigationsbefehle laufen in einem eigenen Thread; ihr Ergebnis (NavResult) löst automatisch
nav_succeeded / nav_failed / nav_aborted / nav_rejected aus. Ungültige Trigger werfen transitions.MachineError.

Weitere Stationskomponenten (Greifer, Ladestation, …) als zusätzliche Zustände/Übergänge ergänzen.
"""
import logging
import time

from transitions import MachineError
from transitions.extensions import LockedMachine

from navigate import NavigationController, NavResult, NavStatus

log = logging.getLogger("station")

MOVING = ["navigating", "docking", "undocking"]


class RobotStation:
    STATES = ["init", "idle", "navigating", "docking", "docked", "undocking", "error", "estop"]

    def __init__(self, nav: NavigationController):
        self.nav = nav
        self.last_result = None  # letztes NavResult
        self.machine = LockedMachine(model=self, states=self.STATES, initial="init",
                                     after_state_change="_log_state", auto_transitions=False)
        t = self.machine.add_transition

        # Start: je nach gemessener Pose
        t("_started", "init", "docked", conditions="_is_docked")
        t("_started", "init", "error", conditions="_is_in_zone")   # in der Zone, aber nicht im Dock
        t("_started", "init", "idle")
        t("_start_failed", "init", "error")

        # Befehle
        t("goto", "idle", "navigating", after="_do_goto")
        t("approach", "idle", "navigating", after="_do_approach")
        t("dock", "idle", "docking", after="_do_dock")
        t("undock", ["docked", "error"], "undocking", conditions="_is_in_zone", after="_do_undock")
        t("stop", MOVING, None, after="_do_abort")                  # interner Übergang: Ergebnis kommt als nav_aborted
        t("estop", "*", "estop", after="_do_estop")
        t("reset", ["error", "estop"], "docked", conditions="_is_docked")
        t("reset", ["error", "estop"], "idle", conditions="_is_free")

        # Ergebnisse der Navigation (aus dem Worker-Thread)
        t("nav_succeeded", "navigating", "idle")
        t("nav_succeeded", "docking", "docked")
        t("nav_succeeded", "undocking", "idle")
        t("nav_aborted", "navigating", "idle")
        t("nav_aborted", ["docking", "undocking"], "error")        # Roboter steht evtl. in der Andockbahn
        t("nav_rejected", ["navigating", "docking"], "idle")       # Befehl unzulässig, nichts bewegt
        t("nav_rejected", "undocking", "docked", conditions="_is_docked")
        t("nav_rejected", "undocking", "error")
        t("nav_failed", "docking", "docked", conditions="_is_docked")
        t("nav_failed", MOVING, "error")

    # ---------- Start ----------
    def start(self):
        res = self.nav.initialize()
        self.last_result = res
        if res.ok:
            self._started()
        else:
            self._start_failed()
        return res

    # ---------- Bedingungen ----------
    def _is_docked(self, *args, **kwargs):
        return self.nav.location == NavigationController.DOCKED

    def _is_in_zone(self, *args, **kwargs):
        return self.nav.location != NavigationController.FREE

    def _is_free(self, *args, **kwargs):
        return self.nav.location == NavigationController.FREE

    # ---------- Aktionen ----------
    def _do_goto(self, x, y, theta=None):
        self.nav.run_async(self.nav.go_to, x, y, theta, callback=self._on_result)

    def _do_approach(self, x, y, angle=0.0, standoff=0.5):
        self.nav.run_async(self.nav.approach, x, y, angle, standoff, callback=self._on_result)

    def _do_dock(self):
        self.nav.run_async(self.nav.dock, callback=self._on_result)

    def _do_undock(self):
        self.nav.run_async(self.nav.undock, callback=self._on_result)

    def _do_abort(self):
        self.nav.abort()

    def _do_estop(self):
        self.nav.emergency_stop()
        log.error("NOT-AUS ausgelöst.")

    def _on_result(self, res: NavResult):
        self.last_result = res
        trigger = {NavStatus.SUCCESS: "nav_succeeded", NavStatus.ABORTED: "nav_aborted",
                   NavStatus.REJECTED: "nav_rejected"}.get(res.status, "nav_failed")
        try:
            self.trigger(trigger)
        except MachineError:
            # z.B. Ergebnis kommt nach estop an -> Zustand bleibt
            log.info(f"Ergebnis {res.status.value} im Zustand '{self.state}' ignoriert.")

    def _log_state(self, *args, **kwargs):
        res = self.last_result
        msg = f" ({res.status.value}: {res.message})" if res and self.state not in MOVING else ""
        log.info(f"[Station] Zustand -> {self.state.upper()}{msg}")

    # ---------- Hilfen ----------
    def wait_until(self, states, timeout=None, poll=0.1):
        """Blockiert, bis einer der Zustände erreicht ist. Rückgabe: erreichter Zustand oder None (Timeout)."""
        states = [states] if isinstance(states, str) else states
        t_end = None if timeout is None else time.time() + timeout
        while self.state not in states:
            if t_end is not None and time.time() > t_end:
                return None
            time.sleep(poll)
        return self.state


# ==========================================
# Beispiel (Simulation)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    nav = NavigationController.create(sim=True, sim_start=(2.2, 0.5, 30.0), seed=1)
    station = RobotStation(nav)
    station.start()
    try:
        station.goto(2.5, -0.5)                      # normales Ziel
        station.wait_until(["idle", "error"])
        station.dock()                               # andocken
        station.wait_until(["docked", "error"])
        station.undock()                             # ausdocken
        station.wait_until(["idle", "error"])
        station.goto(1.0, 0.0)                       # Ziel in der Zone -> REJECTED -> zurück nach idle
        station.wait_until(["idle", "error"])
        time.sleep(0.2)
        log.info(f"Endzustand: {station.state}, {nav.status()}")
    finally:
        nav.shutdown()