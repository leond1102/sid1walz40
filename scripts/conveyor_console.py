"""Interaktive Konsole für das Fließband.
Erlaubt die manuelle Eingabe der Fahrdauer in Sekunden.
"""

from loguru import logger
from conveyor_controller import ConveyorRemote

def main() -> None:
    remote = ConveyorRemote(client_id="conveyor_console")
    try:
        remote.connect()
    except OSError as e:
        logger.error(f"Broker nicht erreichbar: {e}")
        raise SystemExit(1)

    print("\n" + "="*40)
    print(" FLIESSBAND KONSOLE AKTIV")
    print("="*40)
    print(" - Zahl eingeben (z.B. 3.5) = Fließband fährt für diese Dauer")
    print(" - 's' eingeben             = Fahrt sofort stoppen")
    print(" - 'q' eingeben             = Konsole beenden")
    print("="*40 + "\n")

    try:
        while True:
            # Wartet auf eine Eingabe des Benutzers (blockiert nicht den MQTT-Hintergrundthread)
            user_input = input("> ").strip().lower()

            if user_input == 'q':
                break
            elif user_input == 's':
                remote.stop()
            elif user_input:
                try:
                    # Erlaubt Punkt und Komma für Dezimalzahlen
                    seconds = float(user_input.replace(',', '.'))
                    remote.drive(seconds)
                except ValueError:
                    print("Ungültige Eingabe! Bitte eine gültige Zahl, 's' oder 'q' eingeben.")
                    
    except KeyboardInterrupt:
        print("\nAbbruch durch STRG+C.")
    finally:
        remote.disconnect()
        logger.info("Konsole beendet.")

if __name__ == "__main__":
    main()