"""
paths.py – wo alle Daten- und Modelldateien liegen (unabhängig vom aktuellen Arbeitsordner).

    models/                 aktive Modelle (von main.py / navigate.py geladen)
      dynamics_model.joblib   Laufmodell          (train_dynamics.py)
      rotation_model.joblib   Drehmodell          (train_rotation.py)
      best.pt                 YOLO-Modell der Stützenerkennung
      kandidaten/             frisch trainierte Modelle, noch nicht übernommen (trainer.py)
      archiv/                 ersetzte Modelle
    data/
      datasets/             gesammelte Trainingsdaten (collect_*.py): *_dataset.csv, *_return_log.csv
        meta/                 Einstellungen jedes Sammellaufs (*_meta_<zeit>.json)
      nav_logs/             Bewegungslogs von navigate.py (nav_log.csv, nav_rotation_log.csv, *_sim.csv)
      plots/                Diagramme aus dem Training
      archiv/               alte Daten, nicht mehr zum Training benutzen

Die Ordner werden beim Import angelegt.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = ROOT / "models"
CANDIDATES = MODELS / "kandidaten"
MODEL_ARCHIVE = MODELS / "archiv"
DYNAMICS_MODEL = MODELS / "dynamics_model.joblib"
ROTATION_MODEL = MODELS / "rotation_model.joblib"

DATA = ROOT / "data"
DATASETS = DATA / "datasets"
META = DATASETS / "meta"
NAV_LOGS = DATA / "nav_logs"
PLOTS = DATA / "plots"
DATA_ARCHIVE = DATA / "archiv"

for _folder in (MODELS, CANDIDATES, MODEL_ARCHIVE, DATASETS, META, NAV_LOGS, PLOTS, DATA_ARCHIVE):
    _folder.mkdir(parents=True, exist_ok=True)


def dataset(name: str) -> str:
    """Pfad einer Datensatzdatei in data/datasets (z. B. dataset("linear_dataset.csv"))."""
    return str(DATASETS / Path(name).name)


def meta_file(csv_path: str, stamp: int) -> str:
    """Pfad der Metadatei eines Sammellaufs: data/datasets/meta/<name>_meta_<zeit>.json."""
    return str(META / f"{Path(csv_path).stem}_meta_{stamp}.json")


def find_data(name: str) -> str:
    """Findet eine CSV auch nur über ihren Namen (alte Aufrufe wie --csv linear_dataset.csv).

    Reihenfolge: wie angegeben, data/datasets/, data/nav_logs/. Sonst unverändert zurück
    (dann meldet das Laden den fehlenden Pfad).
    """
    if Path(name).is_file():
        return name
    for folder in (DATASETS, NAV_LOGS):
        candidate = folder / Path(name).name
        if candidate.is_file():
            return str(candidate)
    return name
