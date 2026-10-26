"""Finalise teacher data: dedup, leakage guard, output file and filter-stats report."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from sensei_sql.teacher import filters as F
from sensei_sql.teacher.pipeline import SOURCES, WorkFiles, assert_no_dev_leakage, db_path

STAGES = {
    "synth": [F.PARSE, F.EXECUTE, F.EMPTY, F.CONSISTENCY, F.DUPLICATE],
    "relabel": [F.PARSE, F.EXECUTE, F.EMPTY, F.GOLD, F.DUPLICATE],
}
OUTPUT_FIELDS = ("db_id", "question", "sql", "source", "pattern", "consistency_score", "hardness")


def spider_style(sql: str) -> str:
    """Rewrite SQL into the subset Spider's hardness parser understands: explicit ``AS`` for
    table aliases and no column aliases (each alias reference is replaced by its expression)."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="sqlite")
    for select in tree.find_all(exp.Select):
        aliases = {}
        for i, proj in enumerate(select.expressions):
            if isinstance(proj, exp.Alias):
                aliases[proj.alias.lower()] = proj.this
                select.expressions[i] = proj.this
        for col in select.find_all(exp.Column):
            if not col.table and col.name.lower() in aliases:
                col.replace(aliases[col.name.lower()].copy())
    return tree.sql(dialect="sqlite")


def _hardness(c: dict) -> str:
    """Spider hardness of the pair's SQL. Relabelled pairs fall back to the gold SQL they
    were verified against (same result) if the teacher SQL can't be parsed."""
    from sensei_sql.eval.testsuite import hardness

    path = db_path(c["db_id"])
    attempts = [lambda: hardness(path, c["sql"]), lambda: hardness(path, spider_style(c["sql"]))]
    if c.get("gold_sql"):
        attempts.append(lambda: hardness(path, c["gold_sql"]))
    for attempt in attempts:
        try:
            return attempt()
        except Exception:  # test-suite's parser rejects much valid SQLite
            continue
    return "unparsed"


def stage_counts(cands: list[dict], stages: list[str]) -> list[dict]:
    """Candidates entering / passing each stage, in order."""
    failed = Counter(c["failed"] for c in cands)
    remaining = len(cands)
    rows = []
    for s in stages:
        passed = remaining - failed[s]
        rows.append(
            {"stage": s, "in": remaining, "passed": passed, "pass_rate": round(passed / remaining, 4) if remaining else None}
        )
        remaining = passed
    return rows


def finalize(out_dir: Path, cfg: dict) -> dict:
    by_source = {s: WorkFiles(out_dir, s).read() for s in SOURCES}
    every = [c for cands in by_source.values() for c in cands]
    assert_no_dev_leakage({c["db_id"] for c in every})

    # Dedup across sources; human (relabel) questions first so they win ties.
    for c in every:
        if c["failed"] == F.DUPLICATE:
            c["failed"] = None
    F.deduplicate(by_source["relabel"] + by_source["synth"], cfg["filters"]["max_questions_per_sql"])

    kept = [c for c in by_source["relabel"] + by_source["synth"] if c["failed"] is None]
    for c in kept:
        c["hardness"] = _hardness(c)
    assert_no_dev_leakage({c["db_id"] for c in kept})  # leakage guard on the final output

    out_file = out_dir / "teacher_pairs.jsonl"
    out_file.write_text("".join(json.dumps({k: c[k] for k in OUTPUT_FIELDS}, ensure_ascii=False) + "\n" for c in kept))

    rng = random.Random(cfg["seed"])
    stats = {
        "output": str(out_file),
        "kept": len(kept),
        "kept_by_source": dict(Counter(c["source"] for c in kept)),
        "stages": {s: stage_counts(by_source[s], STAGES[s]) for s in SOURCES if by_source[s]},
        "kept_by_pattern": dict(Counter(c["pattern"] or "relabel" for c in kept).most_common()),
        "kept_by_hardness": dict(Counter(c["hardness"] for c in kept).most_common()),
        "pairs_per_db": dict(sorted(Counter(c["db_id"] for c in kept).items())),
        "databases": len({c["db_id"] for c in kept}),
        "samples": [
            {k: c[k] for k in OUTPUT_FIELDS}
            for c in rng.sample(kept, min(cfg["report"]["num_samples"], len(kept)))
        ],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "stats.md").write_text(to_markdown(stats))
    return stats


def to_markdown(stats: dict) -> str:
    lines = [
        "# Teacher data: filter report",
        "",
        f"**Kept pairs:** {stats['kept']} across {stats['databases']} databases "
        f"({', '.join(f'{k}: {v}' for k, v in stats['kept_by_source'].items())})",
        "",
    ]
    for source, rows in stats["stages"].items():
        lines += [f"## {source}: pass rate per stage", "", "| stage | in | passed | pass rate |", "|---|---|---|---|"]
        lines += [f"| {r['stage']} | {r['in']} | {r['passed']} | {r['pass_rate']} |" for r in rows]
        lines.append("")
    for title, key in (("By pattern", "kept_by_pattern"), ("By hardness", "kept_by_hardness")):
        lines += [f"## {title}", "", "| | pairs |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in stats[key].items()]
        lines.append("")
    per_db = sorted(stats["pairs_per_db"].values())
    if per_db:
        lines += [
            "## Pairs per database",
            "",
            f"min {per_db[0]}, median {per_db[len(per_db) // 2]}, max {per_db[-1]} (full list in stats.json)",
            "",
        ]
    lines += ["## Random samples", ""]
    for s in stats["samples"]:
        tag = s["pattern"] or "relabel"
        lines.append(f"- **[{s['db_id']} · {tag} · {s['hardness']}]** {s['question']}  \n  `{s['sql']}`")
    return "\n".join(lines) + "\n"
