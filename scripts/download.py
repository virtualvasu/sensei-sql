"""Fetch everything the project needs so later steps run fully offline.

Usage:
    python scripts/download.py [--skip-models] [--skip-spider] [--skip-eval]

Downloads:
  * Spider 1.0 (spider_data.zip)     -> data/spider/
  * teacher + student weights         -> models/<local_dir>/  (pinned in configs/models.yaml)
  * taoyds/test-suite-sql-eval        -> third_party/test-suite-sql-eval/
  * NLTK punkt tokenizer (used by it)  -> ~/nltk_data/ (NLTK refuses group-writable dirs)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensei_sql.paths import CONFIGS, DATA, MODELS, SPIDER, THIRD_PARTY  # noqa: E402

# Official Spider 1.0 release (yale-lily.github.io/spider), hosted on Google Drive.
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
SPIDER_FILES = ["tables.json", "train_spider.json", "train_others.json", "dev.json"]

TEST_SUITE_REPO = "https://github.com/taoyds/test-suite-sql-eval.git"


def download_spider() -> None:
    if all((SPIDER / f).exists() for f in SPIDER_FILES) and (SPIDER / "database").is_dir():
        print(f"[spider] already present at {SPIDER}")
        return
    import gdown

    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "spider_data.zip"
    if not zip_path.exists():
        print("[spider] downloading spider_data.zip ...")
        gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)

    print("[spider] extracting ...")
    tmp = DATA / "_spider_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    # The archive contains a single top-level folder (spider_data/ or spider/).
    (inner,) = [p for p in tmp.iterdir() if p.is_dir() and not p.name.startswith("__")]
    shutil.rmtree(SPIDER, ignore_errors=True)
    shutil.move(str(inner), SPIDER)
    shutil.rmtree(tmp)
    missing = [f for f in SPIDER_FILES if not (SPIDER / f).exists()]
    if missing:
        sys.exit(f"[spider] archive is missing expected files: {missing}")
    print(f"[spider] ready at {SPIDER}")


def download_models() -> None:
    from huggingface_hub import snapshot_download

    cfg = yaml.safe_load((CONFIGS / "models.yaml").read_text())
    for role, spec in cfg.items():
        target = MODELS / spec["local_dir"]
        print(f"[models] {role}: {spec['repo_id']}@{spec['revision'][:10]} -> {target}")
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=target,
        )


def download_test_suite() -> None:
    target = THIRD_PARTY / "test-suite-sql-eval"
    if (target / "evaluation.py").exists():
        print(f"[eval] test-suite-sql-eval already present at {target}")
        return
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", TEST_SUITE_REPO, str(target)], check=True)
    rev = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (target / "REVISION").write_text(rev + "\n")
    print(f"[eval] cloned test-suite-sql-eval @ {rev[:10]}")


def download_nltk() -> None:
    import nltk

    # Default location (~/nltk_data): newer NLTK rejects data dirs with group-writable ancestors.
    for pkg in ("punkt", "punkt_tab"):
        assert nltk.download(pkg, quiet=True), f"failed to download NLTK {pkg}"
    print("[eval] NLTK punkt data ready")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-spider", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    # Downloads need the network even if the offline flags from .env are exported.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(var, None)

    if not args.skip_spider:
        download_spider()
    if not args.skip_eval:
        download_test_suite()
        download_nltk()
    if not args.skip_models:
        download_models()
    print("done.")


if __name__ == "__main__":
    main()
