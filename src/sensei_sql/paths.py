"""Canonical project paths, resolved relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
THIRD_PARTY = ROOT / "third_party"

SPIDER = DATA / "spider"
SPIDER_DB = SPIDER / "database"
