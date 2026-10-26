"""Konsolen-Steuerung für den Greifarm (ESP32-C6) über MQTT.

Läuft auf dem Rechner mit dem MQTT-Broker. Sendet Befehle über
arm_controller.py, zeigt alle Ereignisse des ESP32 an und überwacht per
Heartbeat, ob er erreichbar ist. Alle Werte stehen in arm_config.py.

Befehle:
    home | min     Referenzfahrt (setzt den Nullpunkt)
    max            auf maximale Position fahren
    pos <schritte> absolute Position in Schritten
    cm <wert>      absolute Position in Zentimetern
    open | close   Greifer öffnen / schließen
    grip <us>      Greifer auf Servozeit in Mikrosekunden
    stop           sofortiger Halt der Achse
    state          letzten bekannten Zustand anzeigen
    config         arm_config.py neu laden und an den ESP32 senden
    quit           beenden
"""

import threading
import time

from loguru import logger

from arm_controller import ArmController


def watchdog(arm: ArmController, stop_event: threading.Event) -> None:
    """Meldet einmalig, wenn die Lebenszeichen des ESP32 ausbleiben.

    Args:
        arm: Zu überwachende Verbindung.
        stop_event: Wird gesetzt, wenn der Thread enden soll.
    """
    was_online = False
    while not stop_event.wait(2.0):
        if arm.online:
            if not was_online:
                logger.success("ESP32 erreichbar.")
            was_online = True
        elif was_online:
            logger.error("Keine Lebenszeichen vom ESP32 mehr.")
            was_online = False


def parse_number(text: str) -> float | None:
    """Liest eine Zahl ein, Komma und Punkt sind erlaubt ("12,5" wie "12.5")."""
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def handle_input(arm: ArmController, line: str) -> bool:
    """Wertet eine Eingabezeile aus und ruft die passende Controller-Funktion auf.

    Args:
        arm: Verbindung zum Greifarm.
        line: Eingegebene Zeile.

    Returns:
        False, wenn das Programm beendet werden soll, sonst True.
    """
    parts = line.split()
    if not parts:
        return True
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("quit", "exit"):
        return False
    if cmd == "help":
        print(__doc__)
    elif cmd == "state":
        arm.show_state()
    elif cmd == "stop":
        arm.stop()
    elif cmd in ("home", "min"):
        arm.home()
    elif cmd == "max":
        arm.move_max()
    elif cmd == "open":
        arm.gripper_open()
    elif cmd == "close":
        arm.gripper_close()
    elif cmd == "config":
        arm.reload_config()
    elif cmd in ("pos", "cm", "grip"):
        if not args:
            logger.warning(f"'{cmd}' braucht einen Wert, z. B. '{cmd} 1000'.")
            return True
        value = parse_number(args[0].removesuffix("cm"))
        if value is None:
            logger.warning(f"'{args[0]}' ist keine Zahl.")
            return True
        if cmd == "pos":
            arm.move_steps(int(value))
        elif cmd == "cm":
            arm.move_cm(value)
        else:
            arm.gripper_us(int(value))
    else:
        logger.warning(f"Unbekannter Befehl: {cmd}. 'help' zeigt alle Befehle.")
    return True


def main() -> None:
    try:
        arm = ArmController(client_id="arm_console")
    except ValueError as e:
        logger.error(f"Fehler in arm_config.py: {e}")
        raise SystemExit(1)

    try:
        arm.connect()
    except OSError as e:
        logger.error(f"Broker nicht erreichbar: {e}")
        raise SystemExit(1)

    stop_event = threading.Event()
    threading.Thread(target=watchdog, args=(arm, stop_event), daemon=True).start()

    time.sleep(1.0)  # kurz warten, bis der retained Zustand da ist
    print(__doc__)
    arm.show_state()

    try:
        while True:
            if not handle_input(arm, input("arm> ")):
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        arm.disconnect()
        logger.info("Beendet.")


if __name__ == "__main__":
    main()
