"""Teacher data generation over Spider's training databases.

Per chunk of databases:
  synth:   teacher writes question/SQL pairs per SQL pattern -> parse -> execute -> empty cap
           -> self-consistency (N samples must reproduce the SQL's result)
  relabel: teacher answers Spider's train questions -> parse -> execute -> empty cap
           -> gold check (same result as Spider's gold SQL)
Every candidate is appended to work/<source>.jsonl with the first stage it failed (or null),
and the database to work/<source>.done, so an interrupted run resumes at the next database.
``finalize`` deduplicates, enforces the dev-leakage guard and writes teacher_pairs.jsonl.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import yaml

from sensei_sql.core.prompts import (
    PATTERNS,
    answer_messages,
    extract_sql,
    generate_pairs_messages,
    parse_generated_pairs,
)
from sensei_sql.core.schema import format_schema
from sensei_sql.paths import CONFIGS, MODELS, SPIDER, SPIDER_DB
from sensei_sql.teacher import filters as F

SOURCES = ("synth", "relabel")
TRAIN_FILES = ("train_spider.json", "train_others.json")


# --- databases and leakage guard -----------------------------------------------------------


def dev_db_ids() -> set[str]:
    return {e["db_id"] for e in json.loads((SPIDER / "dev.json").read_text())}


def train_questions() -> list[dict]:
    return [e for f in TRAIN_FILES for e in json.loads((SPIDER / f).read_text())]


def train_db_ids() -> list[str]:
    return sorted({e["db_id"] for e in train_questions()})


def assert_no_dev_leakage(db_ids) -> None:
    leaked = set(db_ids) & dev_db_ids()
    assert not leaked, f"dev databases must never be used for distillation data: {sorted(leaked)}"


def db_path(db_id: str) -> Path:
    return SPIDER_DB / db_id / f"{db_id}.sqlite"


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# --- work files ------------------------------------------------------------------------------


class WorkFiles:
    def __init__(self, out_dir: Path, source: str):
        self.dir = out_dir / "work"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.candidates = self.dir / f"{source}.jsonl"
        self.done = self.dir / f"{source}.done"

    def done_dbs(self) -> set[str]:
        return set(self.done.read_text().split()) if self.done.exists() else set()

    def append(self, rows: list[dict], db_ids: list[str]) -> None:
        with self.candidates.open("a") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with self.done.open("a") as f:
            f.writelines(d + "\n" for d in db_ids)

    def read(self) -> list[dict]:
        if not self.candidates.exists():
            return []
        return [json.loads(line) for line in self.candidates.read_text().splitlines() if line]


# --- teacher model ---------------------------------------------------------------------------


class Teacher:
    def __init__(self):
        from vllm import LLM

        spec = yaml.safe_load((CONFIGS / "models.yaml").read_text())["teacher"]
        self.chat_template_kwargs = spec.get("chat_template_kwargs")
        # Prefix caching is off by default for this hybrid model; turning it on lets the N
        # self-consistency samples and repeated schema prompts share prefill (~1.7x faster).
        self.llm = LLM(
            model=str(MODELS / spec["local_dir"]), seed=0, enable_prefix_caching=True, **spec.get("vllm", {})
        )

    def chat(self, conversations, params) -> list[list[str]]:
        """Returns, per conversation, the list of sampled texts."""
        if not conversations:
            return []
        outs = self.llm.chat(
            conversations, params, use_tqdm=True, chat_template_kwargs=self.chat_template_kwargs
        )
        return [[o.text for o in out.outputs] for out in outs]


def sampling(cfg: dict, seed: int, n: int = 1):
    from vllm import SamplingParams

    return SamplingParams(
        n=n,
        temperature=cfg["temperature"],
        top_p=cfg.get("top_p", 1.0),
        max_tokens=cfg["max_tokens"],
        seed=seed,
    )


# --- shared verification ---------------------------------------------------------------------


def verify_executes(cands: list[dict], fcfg: dict) -> dict[int, object]:
    """Parse + execute stages; returns {index: ExecResult} for candidates that passed."""
    results = {}
    for i, c in enumerate(cands):
        if not F.parses(c["sql"]):
            c["failed"] = F.PARSE
            continue
        res = F.execute(db_path(c["db_id"]), c["sql"], fcfg["exec_timeout_s"], fcfg["max_rows"])
        if not res.ok:
            c["failed"] = F.EXECUTE
            continue
        c["empty"] = F.is_empty(res)
        results[i] = res
    return results


def cap_empty_per_db(cands: list[dict], fcfg: dict, seed: int) -> None:
    by_db: dict[str, list[dict]] = {}
    for c in cands:
        by_db.setdefault(c["db_id"], []).append(c)
    for db_id, group in by_db.items():
        F.cap_empty(group, fcfg["max_empty_fraction"], stable_seed(seed, "empty", db_id))


def self_consistency(teacher: Teacher, cands, results, cfg) -> None:
    scfg, fcfg = cfg["self_consistency"], cfg["filters"]
    todo = [i for i in results if cands[i]["failed"] is None]
    convs = [
        answer_messages(format_schema(db_path(cands[i]["db_id"])), cands[i]["question"]) for i in todo
    ]
    params = [
        sampling(scfg, stable_seed(cfg["seed"], "sc", cands[i]["db_id"], cands[i]["question"]), scfg["num_samples"])
        for i in todo
    ]
    for i, samples in zip(todo, teacher.chat(convs, params)):
        c, ref = cands[i], results[i]
        agree = 0
        for text in samples:
            sql = extract_sql(text)
            if sql and F.parses(sql):
                res = F.execute(db_path(c["db_id"]), sql, fcfg["exec_timeout_s"], fcfg["max_rows"])
                agree += F.same_result(c["sql"], ref, res)
        c["consistency_score"] = round(agree / len(samples), 3)
        if agree < scfg["min_agree"]:
            c["failed"] = F.CONSISTENCY


def gold_check(cands, results, fcfg) -> None:
    for i, res in results.items():
        c = cands[i]
        if c["failed"] is not None:
            continue
        gold = F.execute(db_path(c["db_id"]), c["gold_sql"], fcfg["exec_timeout_s"], fcfg["max_rows"])
        if not (gold.ok and F.same_result(c["gold_sql"], gold, res)):
            c["failed"] = F.GOLD


# --- sources ---------------------------------------------------------------------------------


def run_synth_chunk(teacher: Teacher, db_ids: list[str], cfg: dict) -> list[dict]:
    syn = cfg["synthesis"]
    jobs = [
        (db, pattern, r)
        for db in db_ids
        for pattern in PATTERNS
        for r in range(syn["requests_per_pattern"])
    ]
    convs = [generate_pairs_messages(format_schema(db_path(db)), p, syn["pairs_per_request"]) for db, p, _ in jobs]
    params = [sampling(syn, stable_seed(cfg["seed"], "synth", db, p, r)) for db, p, r in jobs]
    cands = []
    for (db, pattern, r), (text,) in zip(jobs, teacher.chat(convs, params)):
        for pair in parse_generated_pairs(text):
            cands.append(
                {
                    "db_id": db,
                    "question": pair["question"],
                    "sql": pair["sql"],
                    "source": "synth",
                    "pattern": pattern,
                    "request": r,
                    "consistency_score": None,
                    "empty": None,
                    "failed": None,
                }
            )
    results = verify_executes(cands, cfg["filters"])
    cap_empty_per_db(cands, cfg["filters"], cfg["seed"])
    self_consistency(teacher, cands, results, cfg)
    return cands


def run_relabel_chunk(teacher: Teacher, db_ids: list[str], cfg: dict) -> list[dict]:
    rel = cfg["relabel"]
    wanted = set(db_ids)
    questions = [e for e in train_questions() if e["db_id"] in wanted]
    convs = [answer_messages(format_schema(db_path(e["db_id"])), e["question"]) for e in questions]
    cands = [
        {
            "db_id": e["db_id"],
            "question": " ".join(e["question"].split()),
            "sql": extract_sql(text),
            "gold_sql": e["query"],
            "source": "relabel",
            "pattern": None,
            "consistency_score": None,
            "empty": None,
            "failed": None,
        }
        for e, (text,) in zip(questions, teacher.chat(convs, sampling(rel, cfg["seed"])))
    ]
    results = verify_executes(cands, cfg["filters"])
    cap_empty_per_db(cands, cfg["filters"], cfg["seed"])
    gold_check(cands, results, cfg["filters"])
    return cands


RUNNERS = {"synth": run_synth_chunk, "relabel": run_relabel_chunk}


def generate(out_dir: Path, db_ids: list[str], sources: list[str], cfg: dict) -> None:
    assert_no_dev_leakage(db_ids)
    teacher = None
    for source in sources:
        work = WorkFiles(out_dir, source)
        todo = [d for d in db_ids if d not in work.done_dbs()]
        print(f"[teacher] {source}: {len(db_ids) - len(todo)}/{len(db_ids)} databases already done", flush=True)
        step = cfg["dbs_per_chunk"]
        for k in range(0, len(todo), step):
            chunk = todo[k : k + step]
            teacher = teacher or Teacher()
            t0 = time.time()
            cands = RUNNERS[source](teacher, chunk, cfg)
            work.append(cands, chunk)
            kept = sum(c["failed"] is None for c in cands)
            done = len(db_ids) - len(todo) + k + len(chunk)
            print(
                f"[teacher] {source}: {done}/{len(db_ids)} dbs | chunk {len(cands)} candidates, "
                f"{kept} kept, {time.time() - t0:.0f}s",
                flush=True,
            )


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIGS / "teacher.yaml").read_text())
