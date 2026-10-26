#!/usr/bin/env bash
# Startet die Station mit Frontend (Desktop-Symbol und Autostart rufen dieses Skript auf).
# Läuft die Station schon, wird nur das Frontend-Fenster wieder geöffnet.
# Weitere Argumente gehen an main.py, z. B. ./start_station.sh --sim-nav
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=.venv/bin/python

PORT=$($PY -c "import sys; sys.path.insert(0, 'scripts'); from station_frontend import load_config; from pathlib import Path; print(load_config(Path('config/frontend.yaml'))['server']['port'])" 2>/dev/null)
URL="http://localhost:${PORT:-8080}/"

if curl -s -o /dev/null --max-time 2 "${URL}api/config"; then
    echo "Station läuft bereits – öffne nur das Frontend: $URL"
    $PY -c "import sys; sys.path.insert(0, 'scripts'); from station_frontend import open_browser; open_browser('$URL')"
    sleep 3
    exit 0
fi

$PY main.py "$@"
code=$?
if [ $code -ne 0 ]; then
    echo
    echo "main.py wurde mit Fehler $code beendet (Details in logs/)."
    read -r -p "Enter drücken zum Schließen ..."
fi
exit $code
