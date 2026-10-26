#!/usr/bin/env bash
# Desktop-Symbol und Autostart der Station einrichten.
#
#   ./station_desktop.sh icon            Symbole „Station“ und „Bewegungstraining“ auf den Schreibtisch legen
#   ./station_desktop.sh autostart-on    Station nach dem Anmelden automatisch starten
#                                        (mit automatischer Anmeldung = beim Hochfahren)
#   ./station_desktop.sh autostart-off   Autostart wieder ausschalten
#   ./station_desktop.sh status          zeigt, was eingerichtet ist
#
# Die Station läuft in einem Terminalfenster (Meldungen, Befehle); das Frontend öffnet sich im Vollbild.
set -e
ROOT="$(dirname "$(readlink -f "$0")")"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
ICON_FILE="$DESKTOP_DIR/station.desktop"
TRAINER_ICON_FILE="$DESKTOP_DIR/bewegungstraining.desktop"
AUTOSTART_FILE="$HOME/.config/autostart/station.desktop"
AUTOSTART_DELAY_S=10   # nach dem Anmelden so lange warten (Netzwerk, Kameras)

desktop_entry() {   # $1 = zusätzliche Zeilen
    cat <<ENTRY
[Desktop Entry]
Type=Application
Version=1.0
Name=Station
Comment=Station mit Frontend starten (läuft sie schon: Frontend öffnen)
Exec=gnome-terminal --title=Station -- "$ROOT/start_station.sh"
Path=$ROOT
Icon=$ROOT/assets/station.svg
Terminal=false
Categories=Utility;
$1
ENTRY
}

trainer_entry() {
    cat <<ENTRY
[Desktop Entry]
Type=Application
Version=1.0
Name=Bewegungstraining
Comment=Neue Bewegungen des SID antrainieren (unabhängig von der Station)
Exec=gnome-terminal --title=Bewegungstraining -- "$ROOT/start_trainer.sh"
Path=$ROOT
Icon=$ROOT/assets/trainer.svg
Terminal=false
Categories=Utility;
ENTRY
}

trust() {   # GNOME-Schreibtisch (Ubuntu): Starten erlauben, sonst fragt er bei jedem Doppelklick
    chmod +x "$1"
    env -u GIO_MODULE_DIR -u GTK_PATH -u LD_LIBRARY_PATH gio set "$1" metadata::trusted true 2>/dev/null || true
}

case "${1:-}" in
    icon)
        mkdir -p "$DESKTOP_DIR"
        desktop_entry "" > "$ICON_FILE"
        trust "$ICON_FILE"
        trainer_entry > "$TRAINER_ICON_FILE"
        trust "$TRAINER_ICON_FILE"
        echo "Symbole angelegt: $ICON_FILE, $TRAINER_ICON_FILE"
        ;;
    autostart-on)
        mkdir -p "$(dirname "$AUTOSTART_FILE")"
        desktop_entry "X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=$AUTOSTART_DELAY_S" > "$AUTOSTART_FILE"
        echo "Autostart eingeschaltet: $AUTOSTART_FILE"
        if ! grep -qs "^AutomaticLoginEnable *= *[Tt]rue" /etc/gdm3/custom.conf; then
            echo "Hinweis: Startet erst nach dem Anmelden. Für Start beim Hochfahren automatische Anmeldung"
            echo "einschalten (Einstellungen > Benutzer > Automatische Anmeldung)."
        fi
        ;;
    autostart-off)
        rm -f "$AUTOSTART_FILE"
        echo "Autostart ausgeschaltet."
        ;;
    status)
        [ -f "$ICON_FILE" ] && echo "Desktop-Symbol Station: ja ($ICON_FILE)" || echo "Desktop-Symbol Station: nein"
        [ -f "$TRAINER_ICON_FILE" ] && echo "Desktop-Symbol Bewegungstraining: ja" || echo "Desktop-Symbol Bewegungstraining: nein"
        [ -f "$AUTOSTART_FILE" ] && echo "Autostart: an" || echo "Autostart: aus"
        grep -qs "^AutomaticLoginEnable *= *[Tt]rue" /etc/gdm3/custom.conf \
            && echo "Automatische Anmeldung: an (Autostart = Start beim Hochfahren)" \
            || echo "Automatische Anmeldung: aus (Autostart erst nach dem Anmelden)"
        ;;
    *)
        sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
