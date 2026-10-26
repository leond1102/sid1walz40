#!/usr/bin/env bash
# Startet das Bewegungstraining (Desktop-Symbol „Bewegungstraining“).
# Läuft es schon, wird nur das Frontend-Fenster wieder geöffnet.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=.venv/bin/python
URL="http://localhost:8081/"

if curl -s -o /dev/null --max-time 2 "${URL}api/config"; then
    echo "Bewegungstraining läuft bereits – öffne nur das Frontend: $URL"
    $PY -c "import sys; sys.path.insert(0, 'scripts'); from station_frontend import open_browser; open_browser('$URL')"
    sleep 3
    exit 0
fi

$PY trainer.py "$@"
code=$?
if [ $code -ne 0 ]; then
    echo
    echo "trainer.py wurde mit Fehler $code beendet (Details in logs/)."
    read -r -p "Enter drücken zum Schließen ..."
fi
exit $code
