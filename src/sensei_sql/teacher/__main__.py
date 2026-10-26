"""Generate the execution-verified distillation dataset with the teacher.

    python -m sensei_sql.teacher --limit-dbs 5 --out data/distill/pilot   # pilot
    python -m sensei_sql.teacher                                          # all 146 train DBs
    python -m sensei_sql.teacher --finalize-only                          # re-run dedup + report

Resumable: re-running skips databases already recorded in <out>/work/<source>.done.
Outputs <out>/teacher_pairs.jsonl, stats.json and stats.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensei_sql.paths import DATA
from sensei_sql.teacher.pipeline import SOURCES, generate, load_config, train_db_ids
from sensei_sql.teacher.report import finalize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "distill")
    ap.add_argument("--dbs", help="comma-separated db_ids (default: all training databases)")
    ap.add_argument("--limit-dbs", type=int, help="first N training databases (sorted by name)")
    ap.add_argument("--sources", default=",".join(SOURCES), help="synth,relabel")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    db_ids = args.dbs.split(",") if args.dbs else train_db_ids()
    if args.limit_dbs:
        db_ids = db_ids[: args.limit_dbs]
    sources = [s for s in args.sources.split(",") if s]
    unknown = set(sources) - set(SOURCES)
    if unknown:
        ap.error(f"unknown sources: {sorted(unknown)}")

    if not args.finalize_only:
        generate(args.out, db_ids, sources, cfg)
    stats = finalize(args.out, cfg)
    print(json.dumps({k: stats[k] for k in ("kept", "kept_by_source", "stages")}, indent=1))
    print(f"[teacher] report: {args.out / 'stats.md'}")


if __name__ == "__main__":
    main()
